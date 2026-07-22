from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


DEFAULT_NATIVE_REPO = os.environ.get("WAN_NATIVE_REPO", "/opt/Wan2.2")
DEFAULT_MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/models/Wan2.2-T2V-A14B")
DEFAULT_OUTPUT_DIR = os.environ.get("WAN_OUTPUT_DIR", "outputs")


@dataclass
class JobSpec:
    prompt: str
    task: str = "t2v-A14B"
    size: str = "1280x720"
    gpus: int = 8
    model_dir: str = DEFAULT_MODEL_DIR
    native_repo: str = DEFAULT_NATIVE_REPO
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int | None = None
    use_prompt_extend: bool = False
    convert_model_dtype: bool = True
    offload_model: bool = True
    t5_cpu: bool = False

    @property
    def job_id(self) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "random" if self.seed is None else str(self.seed)
        return f"wan-{self.task.lower()}-{stamp}-seed-{suffix}"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def normalize_size(value: str) -> str:
    return value.replace("x", "*")


def native_command(spec: JobSpec) -> list[str]:
    generate = str(pathlib.Path(spec.native_repo) / "generate.py")
    base = [
        generate,
        "--task",
        spec.task,
        "--size",
        normalize_size(spec.size),
        "--ckpt_dir",
        spec.model_dir,
        "--prompt",
        spec.prompt,
    ]
    if spec.offload_model:
        base += ["--offload_model", "True"]
    if spec.convert_model_dtype:
        base += ["--convert_model_dtype"]
    if spec.t5_cpu:
        base += ["--t5_cpu"]
    if spec.use_prompt_extend:
        base += ["--use_prompt_extend"]
    if spec.seed is not None:
        base += ["--base_seed", str(spec.seed)]

    if spec.gpus <= 1:
        return ["python", *base]
    return [
        "torchrun",
        f"--nproc_per_node={spec.gpus}",
        *base,
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(spec.gpus),
    ]


def command_string(parts: list[str]) -> str:
    return " ".join(shell_quote(p) if any(ch.isspace() for ch in p) else p for p in parts)


def load_job(path: str) -> JobSpec:
    data = json.loads(pathlib.Path(path).read_text())
    return JobSpec(**data)


def write_manifest(spec: JobSpec, command: list[str]) -> pathlib.Path:
    root = pathlib.Path(spec.output_dir).expanduser() / spec.job_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = asdict(spec)
    manifest.update(
        {
            "job_id": spec.job_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": command,
            "command_string": command_string(command),
            "git_sha": git_sha(),
        }
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command_string(command) + "\n")
    return manifest_path


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def doctor(_: argparse.Namespace) -> int:
    print(f"machine={platform.machine()}")
    print(f"python={sys.executable}")
    print(f"native_repo={DEFAULT_NATIVE_REPO}")
    print(f"native_repo_exists={pathlib.Path(DEFAULT_NATIVE_REPO).exists()}")
    print(f"model_dir={DEFAULT_MODEL_DIR}")
    print(f"model_dir_exists={pathlib.Path(DEFAULT_MODEL_DIR).exists()}")
    print(f"torchrun={shutil.which('torchrun') or ''}")
    print(f"huggingface-cli={shutil.which('huggingface-cli') or ''}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"cuda_device_{i}={torch.cuda.get_device_name(i)}")
    except Exception as exc:
        print(f"torch_error={type(exc).__name__}: {exc}")
        return 1
    return 0


def plan(args: argparse.Namespace) -> int:
    spec = load_job(args.job) if args.job else JobSpec(
        prompt=args.prompt,
        task=args.task,
        size=args.size,
        gpus=args.gpus,
        model_dir=args.model_dir,
        native_repo=args.native_repo,
        output_dir=args.output_dir,
        seed=args.seed,
        use_prompt_extend=args.prompt_extend,
        offload_model=not args.no_offload,
        convert_model_dtype=not args.no_convert_dtype,
        t5_cpu=args.t5_cpu,
    )
    command = native_command(spec)
    print(command_string(command))
    if args.write_manifest:
        print(f"manifest={write_manifest(spec, command)}")
    return 0


def download(args: argparse.Namespace) -> int:
    cmd = ["huggingface-cli", "download", args.model, "--local-dir", args.local_dir]
    print(command_string(cmd))
    if args.run:
        return subprocess.call(cmd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wan", description="WAN enterprise GPU control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor")
    p_doctor.set_defaults(func=doctor)

    p_download = sub.add_parser("download")
    p_download.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B")
    p_download.add_argument("--local-dir", default=DEFAULT_MODEL_DIR)
    p_download.add_argument("--run", action="store_true")
    p_download.set_defaults(func=download)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("prompt", nargs="?", default="")
    p_plan.add_argument("--job", help="load a JSON job spec")
    p_plan.add_argument("--task", default="t2v-A14B")
    p_plan.add_argument("--size", default="1280x720")
    p_plan.add_argument("--gpus", type=int, default=8)
    p_plan.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p_plan.add_argument("--native-repo", default=DEFAULT_NATIVE_REPO)
    p_plan.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_plan.add_argument("--seed", type=int)
    p_plan.add_argument("--prompt-extend", action="store_true")
    p_plan.add_argument("--t5-cpu", action="store_true")
    p_plan.add_argument("--no-offload", action="store_true")
    p_plan.add_argument("--no-convert-dtype", action="store_true")
    p_plan.add_argument("--write-manifest", action="store_true")
    p_plan.set_defaults(func=plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

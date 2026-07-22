from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


DEFAULT_NATIVE_REPO = os.environ.get("WAN_NATIVE_REPO", "/opt/Wan2.2")
DEFAULT_MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/models/Wan2.2-T2V-A14B")
DEFAULT_OUTPUT_DIR = os.environ.get("WAN_OUTPUT_DIR", "outputs")
DEFAULT_STATE_DIR = os.environ.get("WAN_STATE_DIR", ".wand")


@dataclass
class JobSpec:
    prompt: str
    job_id: str | None = None
    task: str = "t2v-A14B"
    size: str = "1280x720"
    gpus: int = 1
    model_dir: str = DEFAULT_MODEL_DIR
    native_repo: str = DEFAULT_NATIVE_REPO
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int | None = None
    use_prompt_extend: bool = False
    convert_model_dtype: bool = True
    offload_model: bool = True
    t5_cpu: bool = False

    def ensure_job_id(self) -> str:
        if self.job_id:
            return self.job_id
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = str(random.randrange(100000, 999999)) if self.seed is None else str(self.seed)
        self.job_id = f"wan-{self.task.lower()}-{stamp}-seed-{suffix}"
        return self.job_id


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def normalize_size(value: str) -> str:
    return value.replace("x", "*")


def native_command(spec: JobSpec) -> list[str]:
    generate = str(pathlib.Path(spec.native_repo) / "generate.py")
    python = sys.executable
    torchrun = str(pathlib.Path(sys.executable).with_name("torchrun"))
    if not pathlib.Path(torchrun).exists():
        torchrun = shutil.which("torchrun") or "torchrun"
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
        return [python, *base]
    return [
        torchrun,
        f"--nproc_per_node={spec.gpus}",
        *base,
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(spec.gpus),
    ]


def command_string(parts: list[str]) -> str:
    special = set(" \t\n*?[];$&(){}<>|\"'`")
    return " ".join(shell_quote(p) if any(ch in special for ch in p) else p for p in parts)


def load_job(path: str) -> JobSpec:
    data = json.loads(pathlib.Path(path).read_text())
    return JobSpec(**data)


def write_manifest(spec: JobSpec, command: list[str]) -> pathlib.Path:
    job_id = spec.ensure_job_id()
    root = pathlib.Path(spec.output_dir).expanduser() / job_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = asdict(spec)
    manifest.update(
        {
            "job_id": job_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": command,
            "command_string": command_string(command),
            "git_sha": git_sha(),
            "status": "planned",
        }
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command_string(command) + "\n")
    return manifest_path


def state_paths(state_dir: str = DEFAULT_STATE_DIR) -> dict[str, pathlib.Path]:
    root = pathlib.Path(state_dir).expanduser()
    return {
        "root": root,
        "queue": root / "queue",
        "running": root / "running",
        "done": root / "done",
        "failed": root / "failed",
    }


def ensure_state_dirs(state_dir: str = DEFAULT_STATE_DIR) -> dict[str, pathlib.Path]:
    paths = state_paths(state_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def spec_from_args(args: argparse.Namespace) -> JobSpec:
    return load_job(args.job) if getattr(args, "job", None) else JobSpec(
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


def queue_job(spec: JobSpec, state_dir: str = DEFAULT_STATE_DIR) -> pathlib.Path:
    paths = ensure_state_dirs(state_dir)
    job_id = spec.ensure_job_id()
    path = paths["queue"] / f"{job_id}.json"
    path.write_text(json.dumps(asdict(spec), indent=2, sort_keys=True) + "\n")
    return path


def job_output_dir(spec: JobSpec) -> pathlib.Path:
    return pathlib.Path(spec.output_dir).expanduser() / spec.ensure_job_id()


def update_manifest(path: pathlib.Path, **updates: object) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_spec(spec: JobSpec, dry_run: bool = False) -> int:
    spec.ensure_job_id()
    command = native_command(spec)
    out_dir = job_output_dir(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(spec, command)
    update_manifest(
        manifest_path,
        status="running",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    if dry_run:
        update_manifest(manifest_path, status="planned")
        print(command_string(command))
        print(f"manifest={manifest_path}")
        return 0

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    started = time.time()
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        try:
            proc = subprocess.run(command, cwd=spec.native_repo, stdout=stdout, stderr=stderr, check=False)
            returncode = proc.returncode
        except OSError as exc:
            stderr.write(f"{type(exc).__name__}: {exc}\n".encode())
            returncode = 127
    elapsed = time.time() - started
    status = "done" if returncode == 0 else "failed"
    update_manifest(
        manifest_path,
        status=status,
        returncode=returncode,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seconds=round(elapsed, 3),
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
    )
    print(f"job_id={spec.job_id}")
    print(f"status={status}")
    print(f"manifest={manifest_path}")
    return returncode


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
    bin_dir = pathlib.Path(sys.executable).parent
    print(f"torchrun={bin_dir / 'torchrun' if (bin_dir / 'torchrun').exists() else shutil.which('torchrun') or ''}")
    print(f"huggingface-cli={bin_dir / 'huggingface-cli' if (bin_dir / 'huggingface-cli').exists() else shutil.which('huggingface-cli') or ''}")
    print(f"hf={bin_dir / 'hf' if (bin_dir / 'hf').exists() else shutil.which('hf') or ''}")
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
    spec = spec_from_args(args)
    command = native_command(spec)
    print(command_string(command))
    if args.write_manifest:
        print(f"manifest={write_manifest(spec, command)}")
    return 0


def enqueue(args: argparse.Namespace) -> int:
    spec = spec_from_args(args)
    path = queue_job(spec, args.state_dir)
    print(f"job_id={spec.job_id}")
    print(f"queued={path}")
    return 0


def run_next(args: argparse.Namespace) -> int:
    paths = ensure_state_dirs(args.state_dir)
    queued = sorted(paths["queue"].glob("*.json"))
    if not queued:
        if not args.quiet:
            print("queue=empty")
        return 0

    queued_path = queued[0]
    running_path = paths["running"] / queued_path.name
    queued_path.replace(running_path)
    spec = load_job(str(running_path))
    rc = run_spec(spec, dry_run=args.dry_run)
    target_dir = paths["done"] if rc == 0 else paths["failed"]
    running_path.replace(target_dir / running_path.name)
    return rc


def worker(args: argparse.Namespace) -> int:
    processed = 0
    print(f"state_dir={args.state_dir}")
    print(f"poll_seconds={args.poll}")
    while True:
        paths = ensure_state_dirs(args.state_dir)
        if not any(paths["queue"].glob("*.json")):
            if args.once:
                return 0
            time.sleep(args.poll)
            continue
        rc = run_next(argparse.Namespace(state_dir=args.state_dir, dry_run=args.dry_run, quiet=True))
        processed += 1
        if rc != 0 and args.stop_on_failure:
            return rc
        if args.max_jobs and processed >= args.max_jobs:
            return 0


def jobs(args: argparse.Namespace) -> int:
    paths = ensure_state_dirs(args.state_dir)
    for name in ("queue", "running", "done", "failed"):
        files = sorted(paths[name].glob("*.json"))
        print(f"{name}={len(files)}")
        if args.verbose:
            for path in files[-args.limit:]:
                print(f"  {path}")
    return 0


def download(args: argparse.Namespace) -> int:
    bin_dir = pathlib.Path(sys.executable).parent
    hf = bin_dir / "hf"
    huggingface_cli = bin_dir / "huggingface-cli"
    if hf.exists():
        cmd = [str(hf), "download", args.model, "--local-dir", args.local_dir]
    elif huggingface_cli.exists():
        cmd = [str(huggingface_cli), "download", args.model, "--local-dir", args.local_dir]
    else:
        cmd = [shutil.which("hf") or shutil.which("huggingface-cli") or "huggingface-cli", "download", args.model, "--local-dir", args.local_dir]
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
    p_plan.add_argument("--gpus", type=int, default=1)
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

    p_enqueue = sub.add_parser("enqueue")
    p_enqueue.add_argument("prompt", nargs="?", default="")
    p_enqueue.add_argument("--job", help="load a JSON job spec")
    p_enqueue.add_argument("--task", default="t2v-A14B")
    p_enqueue.add_argument("--size", default="1280x720")
    p_enqueue.add_argument("--gpus", type=int, default=1)
    p_enqueue.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p_enqueue.add_argument("--native-repo", default=DEFAULT_NATIVE_REPO)
    p_enqueue.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_enqueue.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    p_enqueue.add_argument("--seed", type=int)
    p_enqueue.add_argument("--prompt-extend", action="store_true")
    p_enqueue.add_argument("--t5-cpu", action="store_true")
    p_enqueue.add_argument("--no-offload", action="store_true")
    p_enqueue.add_argument("--no-convert-dtype", action="store_true")
    p_enqueue.set_defaults(func=enqueue)

    p_run_next = sub.add_parser("run-next")
    p_run_next.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    p_run_next.add_argument("--dry-run", action="store_true")
    p_run_next.add_argument("--quiet", action="store_true")
    p_run_next.set_defaults(func=run_next)

    p_worker = sub.add_parser("worker")
    p_worker.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    p_worker.add_argument("--poll", type=float, default=10.0)
    p_worker.add_argument("--once", action="store_true")
    p_worker.add_argument("--dry-run", action="store_true")
    p_worker.add_argument("--max-jobs", type=int, default=0)
    p_worker.add_argument("--stop-on-failure", action="store_true")
    p_worker.set_defaults(func=worker)

    p_jobs = sub.add_parser("jobs")
    p_jobs.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    p_jobs.add_argument("--verbose", "-v", action="store_true")
    p_jobs.add_argument("--limit", type=int, default=20)
    p_jobs.set_defaults(func=jobs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

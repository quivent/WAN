from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import os
import pathlib
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_OUTPUT_DIR = os.environ.get("WAN_OUTPUT_DIR", "/runs/wan/outputs" if pathlib.Path("/runs/wan").exists() else "outputs")
DEFAULT_STATE_DIR = os.environ.get("WAN_STATE_DIR", "/runs/wan/.wand" if pathlib.Path("/runs/wan").exists() else ".wand")
DEFAULT_COUNCIL_ROOT = os.environ.get("COUNCIL_ROOT", str(pathlib.Path.home() / "Council-of-Gemmas"))
DEFAULT_RENDER_ROOT = os.environ.get("RENDER_ROOT", str(pathlib.Path.home() / "render"))
DEFAULT_NEXUS_HOST = os.environ.get("NEXUS_HOST", "127.0.0.1")
DEFAULT_NEXUS_PORT = int(os.environ.get("NEXUS_PORT", "9999"))
DEFAULT_PIPER_SOCKET = os.environ.get("PIPER_SOCKET", "/tmp/piper.sock")

MODEL_PRESETS = {
    "T2V": ("Wan-AI/Wan2.2-T2V-A14B", "/models/Wan2.2-T2V-A14B"),
    "I2V": ("Wan-AI/Wan2.2-I2V-A14B", "/models/Wan2.2-I2V-A14B"),
    "TI2V": ("Wan-AI/Wan2.2-TI2V-5B", "/models/Wan2.2-TI2V-5B"),
}


def valid_native_repo(path: str) -> bool:
    root = pathlib.Path(path).expanduser()
    return (root / "generate.py").is_file()


def resolve_native_repo() -> str:
    candidates = [
        os.environ.get("WAN_NATIVE_REPO", ""),
        "/opt/Wan2.2",
        str(pathlib.Path.home() / "Wan2.2"),
        str(pathlib.Path.home() / "Council-of-Gemmas" / "cli" / "cmd" / "embedded" / "wan-pipeline"),
    ]
    for candidate in candidates:
        if candidate and valid_native_repo(candidate):
            return candidate
    return os.environ.get("WAN_NATIVE_REPO", "/opt/Wan2.2")


def valid_wan_model_dir(path: str) -> bool:
    root = pathlib.Path(path).expanduser()
    if not root.is_dir():
        return False
    markers = [
        root / "Wan2.2_VAE.pth",
        root / "models_t5_umt5-xxl-enc-bf16.pth",
        root / "low_noise_model",
        root / "high_noise_model",
    ]
    return any(marker.exists() for marker in markers) and any(root.rglob("*.safetensors"))


def resolve_wan_model_dir(target: str = "T2V") -> str:
    preset_dir = MODEL_PRESETS.get(str(target or "T2V").upper(), MODEL_PRESETS["T2V"])[1]
    candidates = [
        os.environ.get("WAN_MODEL_DIR", ""),
        preset_dir,
        "/models/Wan2.2-T2V-A14B",
        "/models/Wan2.2-I2V-A14B",
        "/models/Wan2.2-TI2V-5B",
        str(pathlib.Path.home() / "models" / pathlib.Path(preset_dir).name),
    ]
    for candidate in candidates:
        if candidate and valid_wan_model_dir(candidate):
            return candidate
    return os.environ.get("WAN_MODEL_DIR", preset_dir)


DEFAULT_NATIVE_REPO = resolve_native_repo()
DEFAULT_MODEL_DIR = resolve_wan_model_dir()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
VIOLET = "\033[38;5;141m"
INDIGO = "\033[38;5;99m"
TEAL = "\033[38;5;73m"
MINT = "\033[38;5;121m"
GOLD = "\033[38;5;220m"
ROSE = "\033[38;5;204m"
AMBER = "\033[38;5;214m"
SOFT = "\033[38;5;246m"


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
    frame_num: int | None = None
    sample_solver: str | None = None
    sample_steps: int | None = None
    sample_shift: float | None = None
    sample_guide_scale: str | None = None
    save_file: str | None = None
    image: str | None = None
    prompt_extend_method: str | None = None
    prompt_extend_model: str | None = None
    prompt_extend_target_lang: str | None = None
    src_root_path: str | None = None
    refert_num: int | None = None
    replace_flag: bool = False
    use_relighting_lora: bool = False
    num_clip: int | None = None
    audio: str | None = None
    enable_tts: bool = False
    tts_prompt_audio: str | None = None
    tts_prompt_text: str | None = None
    tts_text: str | None = None
    pose_video: str | None = None
    start_from_ref: bool = False
    infer_frames: int | None = None
    extra_args: list[str] = field(default_factory=list)

    def ensure_job_id(self) -> str:
        if self.job_id:
            return self.job_id
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = str(random.randrange(100000, 999999)) if self.seed is None else str(self.seed)
        self.job_id = f"wan-{self.task.lower()}-{stamp}-seed-{suffix}"
        return self.job_id


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def color_enabled() -> bool:
    if os.environ.get("WAN_FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    if os.environ.get("WAN_NO_COLOR") or os.environ.get("NO_COLOR"):
        return False
    return True


def paint(color: str, text: str) -> str:
    if not color_enabled():
        return text
    return f"{color}{text}{RESET}"


def strong(text: str) -> str:
    return paint(BOLD, text)


def soft(text: str) -> str:
    return paint(SOFT, text)


def state_text(value: str) -> str:
    key = str(value).lower()
    if key in {"ready", "present", "active", "done", "ok", "queued"}:
        return paint(BOLD + MINT, value)
    if key in {"planned", "downloading", "running", "pending"}:
        return paint(BOLD + AMBER, value)
    if key in {"missing", "failed", "error", "false"}:
        return paint(BOLD + ROSE, value)
    return paint(BOLD + TEAL, value)


def header(title: str, subtitle: str = "") -> None:
    print()
    line = paint(BOLD + VIOLET, title)
    if subtitle:
        line += paint(DIM, f"  {subtitle}")
    print(line)
    print(paint(INDIGO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))


def kv(name: str, value: object) -> None:
    print(f"  {paint(SOFT, name.upper().ljust(18))} {value}")


def command_block(command: list[str] | str, label: str = "Command") -> None:
    text = command if isinstance(command, str) else command_string(command)
    print()
    print(paint(GOLD, label))
    print(f"  {paint(GOLD, '$ ')}{paint(TEAL + BOLD, text)}")


def suite(name: str, color: str, rows: list[tuple[str, str]]) -> None:
    print(paint(color, "▸ ") + paint(BOLD + color, name))
    for index, (left, right) in enumerate(rows):
        branch = "└─" if index == len(rows) - 1 else "├─"
        print(f"  {paint(color, branch)} {paint(color, left.ljust(30))} {paint(DIM, right)}")


def banner() -> None:
    print()
    print(paint(VIOLET, " __        ___    _   _ "))
    print(paint(INDIGO, " \\ \\      / / \\  | \\ | |"))
    print(paint(TEAL, "  \\ \\ /\\ / / _ \\ |  \\| |"))
    print(paint(MINT, "   \\ V  V / ___ \\| |\\  |"))
    print(paint(GOLD, "    \\_/\\_/_/   \\_\\_| \\_|"))
    print(paint(BOLD + VIOLET, "wan") + paint(DIM, "  enterprise GPU video forge"))
    print(paint(INDIGO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(paint(DIM, "  Native Wan2.2 · H200 queue worker · reproducible video jobs"))


def palette(_: argparse.Namespace | None = None) -> int:
    header("colors", "WAN terminal palette")
    rows = [
        ("violet", VIOLET, "kernel / primary headers"),
        ("indigo", INDIGO, "runtime / rules"),
        ("teal", TEAL, "forge / live command text"),
        ("mint", MINT, "ready / complete / present"),
        ("gold", GOLD, "prompt / synthesis / highlights"),
        ("amber", AMBER, "queued / running / planned"),
        ("rose", ROSE, "error / blocked state"),
        ("ink-dim", SOFT, "descriptions and metadata"),
    ]
    for name, color, use in rows:
        print(f"  {paint(color, '● ' + name).ljust(24)} {paint(DIM, use)}")
    print()
    suite("states", TEAL, [
        ("present", state_text("present")),
        ("ready", state_text("ready")),
        ("queued", state_text("queued")),
        ("running", state_text("running")),
        ("done", state_text("done")),
        ("failed", state_text("failed")),
    ])
    return 0


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
    optional = [
        ("--frame_num", spec.frame_num),
        ("--sample_solver", spec.sample_solver),
        ("--sample_steps", spec.sample_steps),
        ("--sample_shift", spec.sample_shift),
        ("--sample_guide_scale", spec.sample_guide_scale),
        ("--save_file", spec.save_file),
        ("--image", spec.image),
        ("--prompt_extend_method", spec.prompt_extend_method),
        ("--prompt_extend_model", spec.prompt_extend_model),
        ("--prompt_extend_target_lang", spec.prompt_extend_target_lang),
        ("--src_root_path", spec.src_root_path),
        ("--refert_num", spec.refert_num),
        ("--num_clip", spec.num_clip),
        ("--audio", spec.audio),
        ("--tts_prompt_audio", spec.tts_prompt_audio),
        ("--tts_prompt_text", spec.tts_prompt_text),
        ("--tts_text", spec.tts_text),
        ("--pose_video", spec.pose_video),
        ("--infer_frames", spec.infer_frames),
    ]
    for flag, value in optional:
        if value is not None and value != "":
            base += [flag, str(value)]
    if spec.replace_flag:
        base += ["--replace_flag"]
    if spec.use_relighting_lora:
        base += ["--use_relighting_lora"]
    if spec.enable_tts:
        base += ["--enable_tts"]
    if spec.start_from_ref:
        base += ["--start_from_ref"]
    if spec.extra_args:
        base += [str(part) for part in spec.extra_args]

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


def parse_addr(value: str) -> tuple[str, int]:
    if ":" not in value:
        return value, 7862
    host, raw_port = value.rsplit(":", 1)
    return host or "127.0.0.1", int(raw_port)


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


def safe_job_id(value: object) -> str:
    text = "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"-", "_"})
    return text[:140] or f"wan-job-{int(time.time())}"


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
        frame_num=args.frame_num,
        sample_solver=args.sample_solver,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        save_file=args.save_file,
        image=args.image,
        prompt_extend_method=args.prompt_extend_method,
        prompt_extend_model=args.prompt_extend_model,
        prompt_extend_target_lang=args.prompt_extend_target_lang,
        src_root_path=args.src_root_path,
        refert_num=args.refert_num,
        replace_flag=args.replace_flag,
        use_relighting_lora=args.use_relighting_lora,
        num_clip=args.num_clip,
        audio=args.audio,
        enable_tts=args.enable_tts,
        tts_prompt_audio=args.tts_prompt_audio,
        tts_prompt_text=args.tts_prompt_text,
        tts_text=args.tts_text,
        pose_video=args.pose_video,
        start_from_ref=args.start_from_ref,
        infer_frames=args.infer_frames,
        extra_args=args.extra_arg or [],
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
        header("run", "foreground WAN render plan")
        kv("state", state_text("planned"))
        kv("job id", spec.job_id)
        kv("manifest", manifest_path)
        kv("output", out_dir)
        command_block(command)
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
    header("run", "foreground WAN render complete")
    kv("job id", spec.job_id)
    kv("status", state_text(status))
    kv("manifest", manifest_path)
    kv("stdout", stdout_path)
    kv("stderr", stderr_path)
    return returncode


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def nexus_request(payload: dict[str, Any], host: str = DEFAULT_NEXUS_HOST, port: int = DEFAULT_NEXUS_PORT, timeout: float = 2.5) -> dict[str, Any]:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as conn:
            conn.settimeout(timeout)
            conn.sendall((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                try:
                    data = conn.recv(65_536)
                except TimeoutError:
                    break
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
    except Exception as exc:
        return {"schema": "council.nexus/v1", "service": "nexus", "ok": False, "status": "offline", "error": str(exc), "host": host, "port": port}
    text = b"".join(chunks).decode("utf-8", "replace").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"schema": "council.nexus/v1", "service": "nexus", "ok": False, "status": "invalid-response", "response": text}
    return parsed if isinstance(parsed, dict) else {"schema": "council.nexus/v1", "service": "nexus", "ok": False, "status": "invalid-response", "response": text}


def piper_request(payload: dict[str, Any], socket_path: str = DEFAULT_PIPER_SOCKET, timeout: float = 2.5) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(socket_path)
            conn.sendall((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            conn.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                try:
                    data = conn.recv(65_536)
                except TimeoutError:
                    break
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
    except Exception as exc:
        return {"schema": "council.piper/v1", "service": "piper", "ok": False, "status": "offline", "error": str(exc), "socket": socket_path}
    text = b"".join(chunks).decode("utf-8", "replace").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"schema": "council.piper/v1", "service": "piper", "ok": False, "status": "invalid-response", "response": text}
    return parsed if isinstance(parsed, dict) else {"schema": "council.piper/v1", "service": "piper", "ok": False, "status": "invalid-response", "response": text}


def council_available(root: str) -> bool:
    return (pathlib.Path(root).expanduser() / "daemons" / "nexus.py").is_file()


def nexus_job_payload(spec: JobSpec, queued_path: pathlib.Path, state_dir: str) -> dict[str, Any]:
    job_id = spec.ensure_job_id()
    output_dir = str(job_output_dir(spec))
    manifest = {
        "schema": "council.nexus.wan_manifest/v1",
        "pipeline": "wan2.2-t2v",
        "runner": "wan-enterprise-runner",
        "generation": {
            "task": spec.task,
            "size": spec.size,
            "gpus": spec.gpus,
            "seed": spec.seed,
            "prompt_extend": spec.use_prompt_extend,
            "offload_model": spec.offload_model,
            "convert_model_dtype": spec.convert_model_dtype,
            "t5_cpu": spec.t5_cpu,
        },
        "required_assets": [
            {"id": "wan-native-runtime", "path": str(pathlib.Path(spec.native_repo) / "generate.py")},
            {"id": "wan-model", "path": spec.model_dir},
        ],
        "wan_queue_job": {
            "state_dir": state_dir,
            "queue_file": str(queued_path),
            "output_dir": output_dir,
            "command": command_string(native_command(spec)),
        },
    }
    return {
        "schema": "grid.visual.workflow.v1",
        "job_id": job_id,
        "id": job_id,
        "kind": "nexus.wan.t2v",
        "title": "WAN text-to-video render",
        "lane": "wan-video",
        "priority": 70,
        "status": "queued",
        "prompt": spec.prompt,
        "brief": "WAN CLI queued this text-to-video job for the continuous H200 runner.",
        "nexus_manifest": manifest,
        "workflow": {
            "intent": "wan.text_to_video.enterprise_gpu",
            "stages": [
                {"id": "manifest", "kind": "wan.manifest.create", "status": "done"},
                {"id": "queue", "kind": "wan.queue.submit", "depends_on": ["manifest"]},
                {"id": "video-generate", "kind": "video.generate", "model": "Wan2.2", "depends_on": ["queue"]},
                {"id": "asset-return", "kind": "nexus.piper.materialize", "depends_on": ["video-generate"]},
            ],
            "models": {"primary_video_model": "Wan2.2", "policy": "wan-strict"},
            "prompt": {"positive": spec.prompt},
        },
        "source": {"submitted_via": "wan render", "runner": "wan-enterprise-runner"},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def publish_nexus_record(spec: JobSpec, queued_path: pathlib.Path, state_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    council_root = pathlib.Path(args.council_root).expanduser()
    render_root = pathlib.Path(args.render_root).expanduser()
    if not council_available(str(council_root)):
        return {"ok": None, "status": "skipped", "reason": "Council-of-Gemmas not found", "council_root": str(council_root)}
    health = nexus_request({"type": "health", "ts": time.time()}, host=args.nexus_host, port=args.nexus_port, timeout=args.nexus_timeout)
    if not health.get("ok"):
        return {"ok": False, "status": health.get("status") or "offline", "reason": health.get("error") or "Nexus is not reachable", "health": health}
    job_id = spec.ensure_job_id()
    state_root = council_root / "council_os" / "state" / "runtime" / "nexus" / "jobs" / job_id
    queued_root = render_root / "grid" / "jobs" / "queued"
    state_root.mkdir(parents=True, exist_ok=True)
    queued_root.mkdir(parents=True, exist_ok=True)
    job = nexus_job_payload(spec, queued_path, state_dir)
    payload = json.dumps(job, indent=2, sort_keys=True) + "\n"
    nexus_job = state_root / "job.json"
    nexus_manifest = state_root / "manifest.json"
    queued_spec = queued_root / f"{job_id}.json"
    nexus_job.write_text(payload, encoding="utf-8")
    nexus_manifest.write_text(json.dumps(job["nexus_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    queued_spec.write_text(payload, encoding="utf-8")
    submit = nexus_request(
        {"type": "submit", "job": {"job_id": job_id, "remote_path": str(queued_spec), "node": "local", "kind": "nexus.wan.t2v"}},
        host=args.nexus_host,
        port=args.nexus_port,
        timeout=args.nexus_timeout,
    )
    return {
        "ok": bool(submit.get("ok")),
        "status": submit.get("status", "unknown"),
        "job_id": job_id,
        "queued_spec": str(queued_spec),
        "state_dir": str(state_root),
        "submit": submit,
    }


def doctor(_: argparse.Namespace) -> int:
    header("doctor", "WAN runtime readiness")
    kv("machine", platform.machine())
    kv("python", sys.executable)
    kv("native repo", DEFAULT_NATIVE_REPO)
    kv("native state", state_text("present" if pathlib.Path(DEFAULT_NATIVE_REPO).exists() else "missing"))
    kv("model dir", DEFAULT_MODEL_DIR)
    kv("model state", state_text("present" if pathlib.Path(DEFAULT_MODEL_DIR).exists() else "missing"))
    kv("output dir", DEFAULT_OUTPUT_DIR)
    kv("state dir", DEFAULT_STATE_DIR)
    bin_dir = pathlib.Path(sys.executable).parent
    kv("torchrun", bin_dir / "torchrun" if (bin_dir / "torchrun").exists() else shutil.which("torchrun") or "")
    kv("hf cli", bin_dir / "hf" if (bin_dir / "hf").exists() else shutil.which("hf") or "")
    try:
        import torch

        kv("torch", torch.__version__)
        kv("cuda", state_text("ready" if torch.cuda.is_available() else "missing"))
        kv("gpu count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            kv(f"gpu {i}", torch.cuda.get_device_name(i))
    except Exception as exc:
        kv("torch", state_text("error"))
        kv("error", f"{type(exc).__name__}: {exc}")
        return 1
    return 0


def architecture(_: argparse.Namespace) -> int:
    header("architecture", "WAN control plane")
    kv("cli", "wan -> Python command router")
    kv("native", DEFAULT_NATIVE_REPO)
    kv("model", DEFAULT_MODEL_DIR)
    kv("queue", DEFAULT_STATE_DIR)
    kv("worker", "wan worker --state-dir " + DEFAULT_STATE_DIR)
    kv("outputs", DEFAULT_OUTPUT_DIR)
    print()
    suite("request flow", TEAL, [
        ("wan render", "queue a Wan2.2 T2V job for the continuous worker"),
        ("wan render --wait", "queue and watch until done or failed"),
        ("wan render --direct", "run native Wan2.2 in the foreground"),
        ("wan render --plan", "show exact command without submitting"),
        ("wan nexus status", "probe Council Nexus on 127.0.0.1:9999"),
        ("wan piper status", "probe Council Piper on /tmp/piper.sock"),
    ])
    return 0


def studio(_: argparse.Namespace) -> int:
    header("studio", "H200 WAN runtime")
    kv("native repo", DEFAULT_NATIVE_REPO)
    kv("native state", state_text("present" if pathlib.Path(DEFAULT_NATIVE_REPO).exists() else "missing"))
    kv("model dir", DEFAULT_MODEL_DIR)
    kv("model state", state_text("present" if pathlib.Path(DEFAULT_MODEL_DIR).exists() else "missing"))
    kv("state dir", DEFAULT_STATE_DIR)
    kv("output dir", DEFAULT_OUTPUT_DIR)
    nexus = nexus_request({"type": "health", "ts": time.time()}, timeout=1.0)
    kv("nexus", state_text(str(nexus.get("status") or "offline")))
    piper = piper_request({"type": "health", "ts": time.time()}, timeout=1.0)
    kv("piper", state_text(str(piper.get("status") or "offline")))
    print()
    suite("commands", GOLD, [
        ("download T2V", "download the text-to-video checkpoint"),
        ("wan render \"prompt\"", "queue one video job"),
        ("wan render \"prompt\" --wait", "queue and follow completion"),
        ("wan jobs --verbose", "inspect queue files"),
        ("wan worker", "run the queue loop in foreground"),
    ])
    return 0


def runtime_config() -> dict[str, Any]:
    return {
        "paths": {
            "WAN_NATIVE_REPO": DEFAULT_NATIVE_REPO,
            "WAN_MODEL_DIR": DEFAULT_MODEL_DIR,
            "WAN_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
            "WAN_STATE_DIR": DEFAULT_STATE_DIR,
        },
        "defaults": {
            "task": "t2v-A14B",
            "size": "1280x720",
            "gpus": 1,
            "frame_num": "native default unless set",
            "sample_steps": "native default unless set",
            "sample_shift": "native default unless set",
            "sample_guide_scale": "native default unless set",
            "offload_model": True,
            "convert_model_dtype": True,
            "t5_cpu": False,
            "prompt_extend": False,
        },
        "precision": {
            "text_encoder": "bf16 checkpoint: models_t5_umt5-xxl-enc-bf16.pth",
            "model_param_dtype": "bf16 from native Wan2.2 config",
            "convert_model_dtype": "enabled by default; pass --no-convert-dtype to disable",
            "offload_model": "enabled by default; pass --no-offload to disable",
        },
        "env_file": "/etc/wan.env",
    }


def config_cmd(args: argparse.Namespace) -> int:
    cfg = runtime_config()
    if args.json:
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0
    header("config", "WAN runtime defaults and tunables")
    for key, value in cfg["paths"].items():
        kv(key, value)
    print()
    suite("defaults", TEAL, [(key, str(value)) for key, value in cfg["defaults"].items()])
    print()
    suite("precision", GOLD, [(key, str(value)) for key, value in cfg["precision"].items()])
    print()
    suite("configure", INDIGO, [
        ("render flags", "--size --gpus --model-dir --native-repo --output-dir --state-dir --no-offload --no-convert-dtype --t5-cpu"),
        ("custom dimensions", "--frame-num --sample-steps --sample-shift --sample-guide-scale --sample-solver --extra-arg"),
        ("service env", "edit /etc/wan.env, then sudo systemctl restart wan-worker"),
        ("worker user", "wan-worker.service runs as ubuntu so CLI queue writes remain valid"),
    ])
    return 0


def gpu(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        payload["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    if shutil.which("nvidia-smi"):
        query = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(query, capture_output=True, text=True, check=False)
        payload["nvidia_smi"] = [line for line in result.stdout.splitlines() if line.strip()]
        pmon = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True, check=False)
        payload["nvidia_pmon"] = [line for line in pmon.stdout.splitlines() if line.strip()]
    else:
        payload["nvidia_smi"] = []
        payload["nvidia_pmon"] = []
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("cuda_available") else 1
    header("gpu", "WAN runtime GPU view")
    kv("python", sys.executable)
    kv("torch", payload.get("torch", "unknown"))
    kv("cuda", state_text("ready" if payload.get("cuda_available") else "missing"))
    kv("device count", payload.get("cuda_device_count", 0))
    for index, name in enumerate(payload.get("cuda_devices") or []):
        kv(f"gpu {index}", name)
    rows = payload.get("nvidia_smi") or []
    if rows:
        print()
        suite("nvidia-smi", TEAL, [(str(row)[:30].strip(), str(row)[30:].strip()) for row in rows])
    processes = payload.get("nvidia_pmon") or []
    if processes:
        print()
        suite("processes", GOLD, [(str(row)[:30].strip(), str(row)[30:].strip()) for row in processes])
    return 0 if payload.get("cuda_available") else 1


def plan(args: argparse.Namespace) -> int:
    spec = spec_from_args(args)
    command = native_command(spec)
    header("plan", "native Wan2.2 command")
    kv("task", spec.task)
    kv("size", spec.size)
    kv("gpus", spec.gpus)
    kv("model", spec.model_dir)
    kv("native repo", spec.native_repo)
    command_block(command)
    if args.write_manifest:
        kv("manifest", write_manifest(spec, command))
    return 0


def enqueue(args: argparse.Namespace) -> int:
    spec = spec_from_args(args)
    path = queue_job(spec, args.state_dir)
    header("enqueue", "queued WAN render job")
    kv("state", state_text("queued"))
    kv("job id", spec.job_id)
    kv("queue file", path)
    kv("state dir", args.state_dir)
    return 0


def locate_job(job_id: str, state_dir: str) -> tuple[str, pathlib.Path] | None:
    paths = state_paths(state_dir)
    target = safe_job_id(job_id)
    for name in ("running", "queue", "done", "failed"):
        path = paths[name] / f"{target}.json"
        if path.exists():
            return name, path
    for name in ("running", "queue", "done", "failed"):
        matches = sorted(paths[name].glob(f"*{target}*.json"))
        if matches:
            return name, matches[-1]
    return None


def wait_for_job(job_id: str, state_dir: str, poll: float, timeout: float) -> int:
    header("wait", "watching WAN queue")
    kv("job id", job_id)
    kv("state dir", state_dir)
    deadline = time.time() + timeout if timeout > 0 else None
    last_state = ""
    while True:
        found = locate_job(job_id, state_dir)
        state = found[0] if found else "missing"
        if state != last_state:
            kv("state", state_text(state))
            last_state = state
        if state == "done":
            return 0
        if state == "failed":
            return 1
        if deadline is not None and time.time() >= deadline:
            kv("timeout", state_text("failed"))
            return 124
        time.sleep(max(0.5, poll))


def render(args: argparse.Namespace) -> int:
    if not getattr(args, "job", None) and not str(args.prompt or "").strip():
        raise SystemExit("wan render needs a prompt or --job")
    spec = spec_from_args(args)
    spec.ensure_job_id()
    if args.plan:
        command = native_command(spec)
        header("render", "WAN job plan")
        kv("state", state_text("planned"))
        kv("job id", spec.job_id)
        kv("task", spec.task)
        kv("size", spec.size)
        kv("gpus", spec.gpus)
        kv("model", spec.model_dir)
        kv("state dir", args.state_dir)
        kv("output dir", spec.output_dir)
        command_block(command)
        return 0
    if args.direct or args.now:
        header("render", "running Wan2.2 in foreground")
        kv("job id", spec.job_id)
        kv("output dir", job_output_dir(spec))
        return run_spec(spec, dry_run=False)

    queued_path = queue_job(spec, args.state_dir)
    header("render", "queued WAN video job")
    kv("state", state_text("queued"))
    kv("job id", spec.job_id)
    kv("queue file", queued_path)
    kv("state dir", args.state_dir)
    kv("output dir", job_output_dir(spec))

    publish = str(args.nexus).lower()
    if publish != "off":
        receipt = publish_nexus_record(spec, queued_path, args.state_dir, args)
        if receipt.get("ok") or publish == "on":
            kv("nexus", state_text(str(receipt.get("status") or "unknown")))
            if receipt.get("queued_spec"):
                kv("nexus spec", receipt["queued_spec"])
        elif publish == "auto":
            kv("nexus", state_text("skipped") + soft(f"  {receipt.get('reason') or receipt.get('status')}"))
        if publish == "on" and not receipt.get("ok"):
            return 1

    if args.wait:
        return wait_for_job(spec.job_id or "", args.state_dir, args.poll, args.timeout)
    return 0


def run_next(args: argparse.Namespace) -> int:
    paths = ensure_state_dirs(args.state_dir)
    queued = sorted(paths["queue"].glob("*.json"))
    if not queued:
        if not args.quiet:
            header("run-next", "queue check")
            kv("state", state_text("empty"))
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
    header("worker", "continuous WAN queue runner")
    kv("state dir", args.state_dir)
    kv("poll", f"{args.poll}s")
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


def parse_ts(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def duration_label(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    if seconds < 3600:
        return f"{int(round(seconds / 60))}m"
    return f"{seconds / 3600:.1f}h"


def job_duration(job: dict[str, Any]) -> float | None:
    raw = job.get("seconds")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    started = parse_ts(job.get("started_at") or job.get("created_at"))
    finished = parse_ts(job.get("finished_at"))
    if started and finished and finished > started:
        return finished - started
    return None


def output_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    job_id = safe_job_id(spec.get("job_id") or spec.get("id") or "")
    output_dir = pathlib.Path(str(spec.get("output_dir") or DEFAULT_OUTPUT_DIR)).expanduser()
    manifest = output_dir / job_id / "manifest.json"
    return read_json(manifest) if manifest.is_file() else {}


def job_record(path: pathlib.Path, state: str) -> dict[str, Any]:
    spec = read_json(path)
    manifest = output_manifest(spec)
    merged = {**spec, **manifest}
    merged["state"] = state
    merged["state_file"] = str(path)
    merged["job_id"] = safe_job_id(merged.get("job_id") or path.stem)
    merged["output_path"] = str(pathlib.Path(str(merged.get("output_dir") or DEFAULT_OUTPUT_DIR)).expanduser() / merged["job_id"])
    return merged


def duration_estimate(records: list[dict[str, Any]], fallback: float = 900.0) -> float:
    samples = [
        duration for record in records
        for duration in [job_duration(record)]
        if duration and duration > 0 and str(record.get("state")).lower() == "done"
    ]
    if not samples:
        return fallback
    samples = sorted(samples[-12:])
    return samples[len(samples) // 2]


def collect_job_records(paths: dict[str, pathlib.Path], limit: int) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            job_record(path, name)
            for path in sorted(paths[name].glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:limit]
        ]
        for name in ("queue", "running", "done", "failed")
    }


def print_job_record(record: dict[str, Any], estimate: float, queued_ahead: float, verbose: bool) -> float:
    state = str(record.get("state") or "")
    job_id = safe_job_id(record.get("job_id"))
    prompt = str(record.get("prompt") or "").strip()
    started = parse_ts(record.get("started_at") or record.get("created_at"))
    elapsed = max(0.0, time.time() - started) if started and state == "running" else None
    eta = max(0.0, estimate - (elapsed or 0.0)) if state == "running" else queued_ahead + estimate
    duration = job_duration(record)
    facts = [
        str(record.get("task") or "t2v-A14B"),
        str(record.get("size") or ""),
        f"seed {record.get('seed')}" if record.get("seed") is not None else "",
        f"elapsed {duration_label(elapsed)}" if elapsed is not None else "",
        f"eta {duration_label(eta)}" if state in {"queue", "running"} else "",
        f"duration {duration_label(duration)}" if duration is not None else "",
    ]
    print(f"    {paint(TEAL, job_id)} {state_text(state)} {soft(' · '.join(x for x in facts if x))}")
    if prompt:
        print(f"      {prompt[:140]}")
    if record.get("output_path"):
        print(f"      {soft(str(record['output_path']))}")
    if verbose:
        print(f"      {soft(str(record.get('state_file') or ''))}")
        if record.get("error"):
            print(f"      {paint(ROSE, str(record.get('error'))[:220])}")
    return eta if state == "queue" else queued_ahead


def jobs(args: argparse.Namespace) -> int:
    paths = ensure_state_dirs(args.state_dir)
    records = collect_job_records(paths, args.limit)
    all_records = [record for values in records.values() for record in values]
    estimate = duration_estimate(all_records)
    if args.json:
        print(json.dumps({"state_dir": args.state_dir, "estimate_seconds": estimate, "jobs": records}, indent=2, sort_keys=True))
        return 0
    header("jobs", "WAN queue state")
    kv("state dir", args.state_dir)
    kv("duration estimate", duration_label(estimate))
    queued_ahead = 0.0
    for name in ("queue", "running", "done", "failed"):
        rows = records[name]
        kv(name, state_text(str(len(rows))) if rows else "0")
        show_rows = args.verbose or name in {"queue", "running", "failed"} or (name == "done" and args.done)
        if show_rows:
            for record in rows:
                queued_ahead = print_job_record(record, estimate, queued_ahead, args.verbose)
    return 0


MEDIA_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".gif", ".webp"}


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def media_kind(path: pathlib.Path) -> str:
    if path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}:
        return "video"
    return "image"


def collect_media(output_root: pathlib.Path, job_id: str) -> list[dict[str, str]]:
    root = output_root.expanduser() / job_id
    if not root.is_dir():
        return []
    items: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTS:
            rel = path.relative_to(output_root.expanduser())
            items.append({
                "name": path.name,
                "kind": media_kind(path),
                "path": str(path),
                "url": "/outputs/" + "/".join(urllib.parse.quote(part) for part in rel.parts),
            })
    return items[:24]


def collect_gallery_jobs(state_dir: str, output_dir: str, limit: int = 80) -> dict[str, Any]:
    paths = ensure_state_dirs(state_dir)
    output_root = pathlib.Path(output_dir).expanduser()
    jobs_out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for state in ("running", "queue", "done", "failed"):
        files = sorted(paths[state].glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        counts[state] = len(files)
        for path in files[:limit]:
            spec = read_json(path)
            job_id = safe_job_id(spec.get("job_id") or path.stem)
            manifest = read_json(output_root / job_id / "manifest.json")
            status = str(manifest.get("status") or state)
            media = collect_media(output_root, job_id)
            jobs_out.append({
                "id": job_id,
                "status": status,
                "queue_state": state,
                "prompt": spec.get("prompt") or manifest.get("prompt") or "",
                "task": spec.get("task") or manifest.get("task") or "",
                "size": spec.get("size") or manifest.get("size") or "",
                "gpus": spec.get("gpus") or manifest.get("gpus") or 1,
                "seed": spec.get("seed") if spec.get("seed") is not None else manifest.get("seed"),
                "output_dir": str(output_root / job_id),
                "manifest": str(output_root / job_id / "manifest.json"),
                "updated": path.stat().st_mtime if path.exists() else 0,
                "media": media,
                "primary_url": media[0]["url"] if media else "",
                "primary_kind": media[0]["kind"] if media else "",
            })
    jobs_out.sort(key=lambda item: float(item.get("updated") or 0), reverse=True)
    return {"ok": True, "state_dir": state_dir, "output_dir": output_dir, "counts": counts, "jobs": jobs_out[:limit]}


def gallery_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAN gallery</title>
<style>
:root{--bg:#060810;--panel:#0d101d;--panel2:#12182a;--text:#ede6d8;--muted:#aeb5c4;--quiet:#6f788b;--line:rgba(237,230,216,.13);--violet:#b48eff;--teal:#64c8ff;--mint:#8dffbd;--gold:#ffd580;--rose:#ff9fb7;--amber:#ffbf72;--shadow:0 18px 60px rgba(0,0,0,.34)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:linear-gradient(180deg,#060810,#0b0e1b 52%,#111426);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0;-webkit-font-smoothing:antialiased}
body:before{content:"";position:fixed;inset:0;pointer-events:none;background:linear-gradient(90deg,rgba(255,159,183,.08),transparent 28%,rgba(100,200,255,.07) 74%,transparent),linear-gradient(180deg,rgba(255,213,128,.045),transparent 34%)}
a{color:inherit}.top{position:sticky;top:0;z-index:4;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;align-items:end;padding:22px clamp(18px,4vw,44px);background:rgba(6,8,16,.82);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}
.mark{color:var(--gold);font:700 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.12em}.title{margin-top:7px;color:var(--violet);font-size:clamp(30px,5vw,62px);font-weight:800;line-height:.95}.sub{max-width:860px;color:var(--muted);margin-top:8px}.status{display:flex;gap:8px;align-items:center;justify-content:flex-end;color:var(--muted);white-space:nowrap}.dot{width:9px;height:9px;border-radius:50%;background:var(--rose);box-shadow:0 0 18px rgba(255,159,183,.55)}.dot.on{background:var(--mint);box-shadow:0 0 18px rgba(141,255,189,.45)}
main{padding:18px clamp(18px,4vw,44px) 42px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px;margin-bottom:16px}.metric{border:1px solid var(--line);border-radius:8px;background:rgba(13,16,29,.72);padding:12px}.metric b{display:block;color:var(--gold);font-size:22px}.metric span{display:block;color:var(--quiet);font:700 10px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;margin-top:5px}
.jobs{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}.job{min-width:0;border:1px solid var(--line);border-radius:8px;background:rgba(13,16,29,.74);box-shadow:var(--shadow);overflow:hidden}.preview{position:relative;aspect-ratio:16/9;background:#050711;display:grid;place-items:center;color:var(--quiet)}.preview img,.preview video{width:100%;height:100%;object-fit:cover;display:block}.preview .placeholder{padding:18px;text-align:center;color:var(--quiet)}.body{padding:12px}.row{display:flex;align-items:center;gap:8px;justify-content:space-between}.job b{min-width:0;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.state{border:1px solid rgba(255,213,128,.18);border-radius:999px;padding:4px 7px;color:var(--gold);font:700 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;background:rgba(255,213,128,.055);white-space:nowrap}.prompt{margin-top:9px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;min-height:60px}.facts{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.facts span{border:1px solid rgba(100,200,255,.16);border-radius:999px;padding:4px 7px;color:var(--teal);font:700 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;background:rgba(100,200,255,.045)}.links{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.button{display:inline-flex;align-items:center;justify-content:center;min-height:30px;padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:rgba(18,24,42,.8);color:var(--text);text-decoration:none;font-size:12px}.button.primary{border-color:rgba(255,213,128,.28);color:var(--gold)}.empty{border:1px dashed var(--line);border-radius:8px;color:var(--muted);padding:24px;text-align:center;background:rgba(13,16,29,.52)}
@media(max-width:720px){.top{grid-template-columns:1fr}.status{justify-content:flex-start}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.jobs{grid-template-columns:1fr}.title{font-size:36px}}
</style>
</head>
<body>
<div class="top"><div><div class="mark">WAN · enterprise video archive</div><div class="title">Render gallery</div><div class="sub">Live H200 queue and output wall for Wan2.2 jobs. The page streams job state and lights up media as files land in the output directory.</div></div><div class="status"><i id="dot" class="dot"></i><span id="status">connecting</span></div></div>
<main>
<section id="metrics" class="metrics"></section>
<section id="jobs" class="jobs"><div class="empty">Loading gallery</div></section>
</main>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),escAttr=s=>esc(s).replace(/"/g,'&quot;');
function setState(ok,msg){$('dot').classList.toggle('on',!!ok);$('status').textContent=msg}
function metrics(c){c=c||{};$('metrics').innerHTML=['running','queue','done','failed'].map(k=>'<div class="metric"><b>'+esc(c[k]||0)+'</b><span>'+esc(k)+'</span></div>').join('')}
function preview(j){if(j.primary_url&&j.primary_kind==='video')return '<video src="'+escAttr(j.primary_url)+'" muted loop playsinline controls></video>';if(j.primary_url)return '<img src="'+escAttr(j.primary_url)+'" alt="">';return '<div class="placeholder">Waiting for media</div>'}
function jobHTML(j){const facts=[j.task,j.size,(j.gpus?j.gpus+' gpu':''),(j.seed!==null&&j.seed!==undefined?'seed '+j.seed:'')].filter(Boolean);let links='';if(j.primary_url)links+='<a class="button primary" href="'+escAttr(j.primary_url)+'" target="_blank" rel="noreferrer">Open media</a>';if(j.manifest)links+='<a class="button" href="'+escAttr('/outputs/'+j.id+'/manifest.json')+'" target="_blank" rel="noreferrer">Manifest</a>';return '<article class="job"><div class="preview">'+preview(j)+'</div><div class="body"><div class="row"><b title="'+escAttr(j.id)+'">'+esc(j.id)+'</b><span class="state">'+esc(j.queue_state||j.status||'')+'</span></div><div class="prompt">'+esc(j.prompt||'No prompt recorded')+'</div><div class="facts">'+facts.map(x=>'<span>'+esc(x)+'</span>').join('')+'</div><div class="links">'+links+'</div></div></article>'}
function render(data){metrics(data.counts);const jobs=data.jobs||[];$('jobs').innerHTML=jobs.map(jobHTML).join('')||'<div class="empty">No WAN jobs yet.</div>'}
async function load(){const r=await fetch('/api/jobs');const j=await r.json();render(j);setState(true,'snapshot ready')}
function connect(){if(!window.EventSource){load().catch(e=>setState(false,e.message));return}const es=new EventSource('/api/jobs/events');es.addEventListener('jobs',ev=>{try{render(JSON.parse(ev.data));setState(true,'stream live')}catch(_){setState(false,'stream parse error')}});es.onerror=()=>{setState(false,'stream reconnecting')};setTimeout(()=>load().catch(()=>{}),1000)}
connect();
</script>
</body>
</html>
"""


class GalleryHandler(http.server.BaseHTTPRequestHandler):
    state_dir = DEFAULT_STATE_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    event_poll = 2.0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/gallery"}:
            data = gallery_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/jobs":
            self.send_json(collect_gallery_jobs(self.state_dir, self.output_dir))
            return
        if parsed.path == "/api/jobs/events":
            self.stream_jobs()
            return
        if parsed.path.startswith("/outputs/"):
            self.serve_output(parsed.path)
            return
        self.send_json({"ok": False, "error": "not found"}, code=404)

    def stream_jobs(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = ""
        while True:
            payload = collect_gallery_jobs(self.state_dir, self.output_dir)
            data = json.dumps(payload, sort_keys=True)
            if data != last:
                try:
                    self.wfile.write(f"event: jobs\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    return
                last = data
            time.sleep(self.event_poll)

    def serve_output(self, path: str) -> None:
        rel = urllib.parse.unquote(path.removeprefix("/outputs/"))
        root = pathlib.Path(self.output_dir).expanduser().resolve()
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            self.send_json({"ok": False, "error": "invalid output path"}, code=400)
            return
        if not target.is_file():
            self.send_json({"ok": False, "error": "missing output"}, code=404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        with target.open("rb") as src:
            shutil.copyfileobj(src, self.wfile)


def gallery(args: argparse.Namespace) -> int:
    host, port = parse_addr(args.addr)
    handler = type("ConfiguredGalleryHandler", (GalleryHandler,), {
        "state_dir": args.state_dir,
        "output_dir": args.output_dir,
        "event_poll": args.poll,
    })
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/gallery"
    header("gallery", "WAN live output wall")
    kv("url", url)
    kv("state dir", args.state_dir)
    kv("output dir", args.output_dir)
    kv("stream", "/api/jobs/events")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def nexus(args: argparse.Namespace) -> int:
    command = getattr(args, "nexus_command", None) or "status"
    if command in {"status", "health"}:
        result = nexus_request({"type": "health", "ts": time.time()}, host=args.host, port=args.port, timeout=args.timeout)
    elif command == "jobs":
        result = nexus_request({"type": "jobs", "limit": args.limit, "job_id": args.job_id}, host=args.host, port=args.port, timeout=args.timeout)
    else:
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    header("nexus", "Council job daemon")
    kv("status", state_text(str(result.get("status") or "unknown")))
    kv("ok", state_text(str(bool(result.get("ok"))).lower()))
    if result.get("host"):
        kv("host", result.get("host"))
    if result.get("port"):
        kv("port", result.get("port"))
    if result.get("piper_connected") is not None:
        kv("piper", state_text("online" if result.get("piper_connected") else "offline"))
    if result.get("queue_root"):
        kv("queue root", result.get("queue_root"))
    if isinstance(result.get("counts"), dict):
        for key, value in result["counts"].items():
            kv(str(key), value)
    return 0 if result.get("ok") else 1


def piper(args: argparse.Namespace) -> int:
    result = piper_request({"type": "health", "ts": time.time()}, socket_path=args.socket, timeout=args.timeout)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    header("piper", "Council asset materializer")
    kv("status", state_text(str(result.get("status") or "unknown")))
    kv("ok", state_text(str(bool(result.get("ok"))).lower()))
    kv("socket", result.get("socket") or args.socket)
    if result.get("asset_cache"):
        kv("asset cache", result.get("asset_cache"))
    return 0 if result.get("ok") else 1


def resolve_model_target(target: str, model: str, local_dir: str) -> tuple[str, str]:
    key = str(target or "T2V").upper()
    preset = MODEL_PRESETS.get(key)
    if preset is not None:
        default_model, default_dir = preset
    elif "/" in str(target):
        default_model = str(target)
        name = default_model.rstrip("/").split("/")[-1]
        default_dir = f"/models/{name}"
    elif target:
        raise ValueError(f"unknown download target {target!r}; use T2V, I2V, TI2V, or a Hugging Face repo id")
    else:
        default_model, default_dir = MODEL_PRESETS["T2V"]
    return model or default_model, local_dir or default_dir


def download(args: argparse.Namespace) -> int:
    model, local_dir = resolve_model_target(args.target, args.model, args.local_dir)
    bin_dir = pathlib.Path(sys.executable).parent
    hf = bin_dir / "hf"
    huggingface_cli = bin_dir / "huggingface-cli"
    if hf.exists():
        cmd = [str(hf), "download", model, "--local-dir", local_dir]
    elif huggingface_cli.exists():
        cmd = [str(huggingface_cli), "download", model, "--local-dir", local_dir]
    else:
        cmd = [shutil.which("hf") or shutil.which("huggingface-cli") or "huggingface-cli", "download", model, "--local-dir", local_dir]
    target = str(args.target or "T2V").upper()
    header("download", "WAN model weights")
    kv("target", target)
    kv("model", model)
    kv("directory", local_dir)
    if args.plan:
        kv("state", state_text("planned"))
        command_block(f"download {target}", "Run")
        command_block(cmd)
        return 0
    kv("state", state_text("downloading"))
    command_block(cmd)
    return subprocess.call(cmd)


def add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", nargs="?", default="T2V", help="model preset: T2V, I2V, TI2V, or a Hugging Face repo id")
    parser.add_argument("--model", default="", help="override Hugging Face repo id")
    parser.add_argument("--local-dir", default="", help="override download directory")
    parser.add_argument("--plan", action="store_true", help="print the download plan without running it")
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(func=download)


def add_job_args(parser: argparse.ArgumentParser, *, state: bool = False) -> None:
    parser.add_argument("prompt", nargs="?", default="")
    parser.add_argument("--job", help="load a JSON job spec")
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--native-repo", default=DEFAULT_NATIVE_REPO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    if state:
        parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-extend", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--no-convert-dtype", action="store_true")
    parser.add_argument("--frame-num", type=int)
    parser.add_argument("--sample-solver", choices=["unipc", "dpm++"])
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--sample-shift", type=float)
    parser.add_argument("--sample-guide-scale")
    parser.add_argument("--save-file")
    parser.add_argument("--image")
    parser.add_argument("--prompt-extend-method", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt-extend-model")
    parser.add_argument("--prompt-extend-target-lang", choices=["zh", "en"])
    parser.add_argument("--src-root-path")
    parser.add_argument("--refert-num", type=int)
    parser.add_argument("--replace-flag", action="store_true")
    parser.add_argument("--use-relighting-lora", action="store_true")
    parser.add_argument("--num-clip", type=int)
    parser.add_argument("--audio")
    parser.add_argument("--enable-tts", action="store_true")
    parser.add_argument("--tts-prompt-audio")
    parser.add_argument("--tts-prompt-text")
    parser.add_argument("--tts-text")
    parser.add_argument("--pose-video")
    parser.add_argument("--start-from-ref", action="store_true")
    parser.add_argument("--infer-frames", type=int)
    parser.add_argument("--extra-arg", action="append", default=[], help="append a raw native Wan2.2 argument; repeat for each token")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wan", description="WAN enterprise GPU control CLI")
    sub = parser.add_subparsers(dest="command")

    p_doctor = sub.add_parser("doctor")
    p_doctor.set_defaults(func=doctor)

    p_arch = sub.add_parser("architecture", aliases=["arch"])
    p_arch.set_defaults(func=architecture)

    p_studio = sub.add_parser("studio", aliases=["status"])
    p_studio.set_defaults(func=studio)

    p_colors = sub.add_parser("colors", aliases=["theme"])
    p_colors.set_defaults(func=palette)

    p_config = sub.add_parser("config", aliases=["defaults", "vars"])
    p_config.add_argument("--json", action="store_true")
    p_config.set_defaults(func=config_cmd)

    p_gpu = sub.add_parser("gpu", aliases=["gpus", "nvidia"])
    p_gpu.add_argument("--json", action="store_true")
    p_gpu.set_defaults(func=gpu)

    p_gallery = sub.add_parser("gallery", aliases=["view"])
    p_gallery.add_argument("--addr", default="127.0.0.1:7862")
    p_gallery.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    p_gallery.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_gallery.add_argument("--poll", type=float, default=2.0)
    p_gallery.add_argument("--open", action="store_true")
    p_gallery.set_defaults(func=gallery)

    p_download = sub.add_parser("download")
    add_download_args(p_download)

    p_plan = sub.add_parser("plan")
    add_job_args(p_plan)
    p_plan.add_argument("--write-manifest", action="store_true")
    p_plan.set_defaults(func=plan)

    p_enqueue = sub.add_parser("enqueue")
    add_job_args(p_enqueue, state=True)
    p_enqueue.set_defaults(func=enqueue)

    p_render = sub.add_parser("render", aliases=["imagine", "forge"])
    add_job_args(p_render, state=True)
    p_render.add_argument("--plan", action="store_true", help="show the render plan without queueing")
    p_render.add_argument("--direct", action="store_true", help="run native Wan2.2 in the foreground")
    p_render.add_argument("--now", action="store_true", help="alias for --direct")
    p_render.add_argument("--wait", action="store_true", help="wait for the queued job to finish")
    p_render.add_argument("--poll", type=float, default=5.0, help="wait polling interval")
    p_render.add_argument("--timeout", type=float, default=0.0, help="wait timeout in seconds; 0 means no timeout")
    p_render.add_argument("--nexus", choices=["auto", "on", "off"], default="auto", help="publish Council Nexus job record")
    p_render.add_argument("--council-root", default=DEFAULT_COUNCIL_ROOT)
    p_render.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    p_render.add_argument("--nexus-host", default=DEFAULT_NEXUS_HOST)
    p_render.add_argument("--nexus-port", type=int, default=DEFAULT_NEXUS_PORT)
    p_render.add_argument("--nexus-timeout", type=float, default=2.5)
    p_render.set_defaults(func=render)

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

    p_jobs = sub.add_parser("jobs", aliases=["queue", "pipeline"])
    p_jobs.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    p_jobs.add_argument("--verbose", "-v", action="store_true")
    p_jobs.add_argument("--limit", type=int, default=20)
    p_jobs.add_argument("--json", action="store_true")
    p_jobs.add_argument("--done", action="store_true", help="include completed rows in the detailed listing")
    p_jobs.set_defaults(func=jobs)

    p_nexus = sub.add_parser("nexus")
    p_nexus.add_argument("nexus_command", nargs="?", choices=["status", "health", "jobs"], default="status")
    p_nexus.add_argument("--host", default=DEFAULT_NEXUS_HOST)
    p_nexus.add_argument("--port", type=int, default=DEFAULT_NEXUS_PORT)
    p_nexus.add_argument("--timeout", type=float, default=2.5)
    p_nexus.add_argument("--limit", type=int, default=80)
    p_nexus.add_argument("--job-id", default="")
    p_nexus.add_argument("--json", action="store_true")
    p_nexus.set_defaults(func=nexus)

    p_piper = sub.add_parser("piper")
    p_piper.add_argument("piper_command", nargs="?", choices=["status", "health"], default="status")
    p_piper.add_argument("--socket", default=DEFAULT_PIPER_SOCKET)
    p_piper.add_argument("--timeout", type=float, default=2.5)
    p_piper.add_argument("--json", action="store_true")
    p_piper.set_defaults(func=piper)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        banner()
        print()
        suite("kernel", VIOLET, [
            ("doctor", "inspect H200 runtime readiness"),
            ("studio", "show paths, model state, Nexus, and Piper"),
            ("architecture", "show CLI, queue, worker, and daemon flow"),
            ("colors", "show the WAN terminal palette"),
            ("gpu", "show H200/Torch GPU state"),
            ("gallery --open", "start the live output wall"),
            ("download T2V", "download text-to-video weights"),
        ])
        suite("forge", TEAL, [
            ("render \"prompt\"", "queue a Wan2.2 video job"),
            ("render \"prompt\" --wait", "queue and watch until completion"),
            ("render \"prompt\" --direct", "run native Wan2.2 in foreground"),
            ("plan \"prompt\"", "show exact native command"),
            ("imagine / forge", "aliases for render"),
        ])
        suite("runtime", INDIGO, [
            ("jobs --verbose", "inspect WAN queue state"),
            ("gpu", "show GPU memory, utilization, and processes"),
            ("gallery --addr 0.0.0.0:7862", "serve queue and media stream"),
            ("worker", "run continuous H200 queue loop"),
            ("run-next", "claim one queued job"),
            ("nexus status", "probe Council Nexus"),
            ("piper status", "probe Council Piper"),
        ])
        return 0
    args = parser.parse_args(argv)
    return args.func(args)


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="download", description="Download WAN model weights")
    add_download_args(parser)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

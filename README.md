# WAN

<pre style="background: #082F49; color: #38BDF8; border: 1px solid #0284C7; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 13px; line-height: 1.25; overflow-x: auto;">
<span style="color: #38BDF8; font-weight: bold;"> ╔═════════════════════════════════════════════════════════════════════════════════════════╗</span>
<span style="color: #60A5FA; font-weight: bold;"> ║                                                                                         ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   ██╗    ██╗ █████╗ ███╗   ██╗    ██████╗  ██╗    ██╗    ██████╗ ███████╗               ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   ██║    ██║██╔══██╗████╗  ██║    ╚════██╗███║    ██║    ██╔══██╗██╔════╝               ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   ██║ █╗ ██║███████║██╔██╗ ██║     █████╔╝╚██║    ██║    ██║  ██║█████╗                 ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   ██║███╗██║██╔══██║██║╚██╗██║    ██╔═══╝  ██║    ██║    ██║  ██║██╔══╝                 ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   ╚███╔███╔╝██║  ██║██║ ╚████║    ███████╗ ██║    ██║    ██████╔╝██║                    ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║    ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚═╝    ╚═╝    ╚═════╝ ╚═╝                    ║</span>
<span style="color: #38BDF8;"> ║                                                                                         ║</span>
<span style="color: #FBBF24; font-weight: bold;"> ║        ───  V I D E O  D I F F U S I O N  &  M O T I O N  S Y N T H E S I S  ───        ║</span>
<span style="color: #38BDF8;"> ║                                                                                         ║</span>
<span style="color: #38BDF8; font-weight: bold;"> ╠═════════════════════════════════════════════════════════════════════════════════════════╣</span>
<span style="color: #38BDF8;"> ║                                                                                         ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   [PIPELINE SPEC]          </span><span style="color: #E2E8F0;">WAN 2.1 Video Generation & Temporal Frame Interpolation        </span><span style="color: #38BDF8;">║</span>
<span style="color: #38BDF8;"> ║                                                                                         ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   [FRAME SYNTHESIS]        </span><span style="color: #E2E8F0;">Keyframe Prompt t_0 ──► Latent Motion Trajectory (30 FPS)    </span><span style="color: #38BDF8;">║</span>
<span style="color: #E2E8F0;"> ║                                               ──► Synthesized High-Res MP4 Video       </span><span style="color: #38BDF8;">║</span>
<span style="color: #38BDF8;"> ║                                                                                         ║</span>
<span style="color: #60A5FA; font-weight: bold;"> ║   [ACCELERATION]           </span><span style="color: #FBBF24; font-weight: bold;">NVIDIA GH200 Grace Hopper (PyTorch 2.5 + FlashAttention-3)    </span><span style="color: #38BDF8;">║</span>
<span style="color: #38BDF8;"> ║                                                                                         ║</span>
<span style="color: #38BDF8; font-weight: bold;"> ╚═════════════════════════════════════════════════════════════════════════════════════════╝</span>
</pre>


WAN is a private operations repo for running Wan video models on enterprise GPU
hosts. It is an analog to the FLUX repo in ownership pattern: one focused repo
for model setup, job manifests, repeatable commands, runtime checks, and remote
execution hygiene.

It is not a port of FLUX.

## Scope

- Native Wan2.2 command planning for enterprise GPUs.
- Diffusers-compatible layout hooks for later service integration.
- Reproducible job manifests under `jobs/`.
- GPU profiles for single-node and multi-GPU execution.
- Docker and Slurm templates for cluster execution.

## Quick Start

```bash
cd /Users/joshkornreich/WAN
make setup
make doctor
make plan PROMPT="a slow cinematic push through a rainy neon market"
```

## H200 Continuous Worker

On a fresh H200 host, clone this repo to `/opt/WAN`, then run:

```bash
cd /opt/WAN
export WAN_NATIVE_REPO=/opt/Wan2.2
export WAN_MODEL_DIR=/models/Wan2.2-T2V-A14B
export WAN_OUTPUT_DIR=/runs/wan/outputs
export WAN_STATE_DIR=/runs/wan/.wand
scripts/bootstrap_h200.sh
download T2V
wan doctor
```

Queue jobs:

```bash
wan render "a slow cinematic push through a rainy neon market" \
  --task t2v-A14B \
  --size 1280x720 \
  --gpus 1

wan render --job jobs/examples/t2v-720p.json
wan render "a quiet spacecraft crossing a red storm" --wait
wan jobs --verbose
```

Run continuously:

```bash
wan worker --state-dir "$WAN_STATE_DIR" --poll 10
```

For systemd, copy `systemd/wan-worker.service` to
`/etc/systemd/system/wan-worker.service`, adjust paths if needed, then enable
it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now wan-worker
```

For Slurm:

```bash
sbatch slurm/wan-worker-h200.sbatch
```

The first concrete target is native Wan2.2 T2V on CUDA:

```bash
wan plan "a slow cinematic push through a rainy neon market" \
  --task t2v-A14B \
  --size 1280x720 \
  --gpus 8 \
  --model-dir /models/Wan2.2-T2V-A14B
```

## Runtime Model

The native lane wraps the official Wan repository rather than importing it into
this repo. Set:

```bash
export WAN_NATIVE_REPO=/opt/Wan2.2
export WAN_MODEL_DIR=/models/Wan2.2-T2V-A14B
export WAN_OUTPUT_DIR=/runs/wan/outputs
```

For an 8 GPU host, the planned command uses `torchrun` with FSDP and Ulysses.
For a 1 GPU 80 GB host, it uses direct `python generate.py` with offload flags.

## Job Artifact Contract

Each executed job should produce:

```text
outputs/{job_id}/
  manifest.json
  command.sh
  stdout.log
  stderr.log
  video.mp4
  metrics.json
```

The manifest is the source of truth: prompt, task, size, seed, model path,
native repo path, GPU count, command, git SHA, and created timestamp.

## Commands

```bash
wan doctor
wan studio
wan architecture
wan colors
wan gpu
wan gallery --open
download T2V
wan render "prompt"
wan render "prompt" --wait
wan render "prompt" --plan
wan render "prompt" --direct
wan imagine "prompt"
wan forge "prompt"
wan plan "prompt" --task t2v-A14B --size 1280x720 --gpus 8
wan plan --job jobs/examples/t2v-720p.json
wan enqueue "prompt" --task t2v-A14B --size 1280x720 --gpus 1
wan worker
wan jobs --verbose
wan gpu
wan queue
wan gallery --addr 0.0.0.0:7862
wan nexus status
wan nexus jobs
wan piper status
```

On a Council host, `wan render` queues the job for the WAN worker and publishes
a Nexus-compatible job record when Nexus is reachable. Piper receives the
queued-spec materialization request through Nexus.
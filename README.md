# RynnValue: Scaling Robotic Value Foundation Models with Temporal Distance

<p align="center">
&nbsp;<a href="https://alibaba-damo-academy.github.io/RynnValue.github.io">🏠 Homepage</a>&nbsp; | &nbsp;<a href="https://github.com/alibaba-damo-academy/RynnValue">💻 GitHub</a>&nbsp; | &nbsp;<a href="https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnvalue">🤗 HuggingFace</a>&nbsp; | &nbsp;<a href="https://www.modelscope.cn/collections/DAMO_Academy/RynnValue">🔮 ModelScope</a>&nbsp; | &nbsp;<a href="https://arxiv.org/abs/2608.09853">📄 ArXiv</a>&nbsp;
</p>

**A general-purpose value foundation model for robot manipulation — and the full toolchain to evaluate it and use it for reinforcement learning of VLA policies.**

RynnValue is a RynnBrain-based vision-language model (implemented on the Qwen3-VL architecture) that watches a robot video together with a task instruction and predicts, **for every frame**, the **temporal distance** — the *remaining time (in seconds) until the task is completed* — alongside a natural-language analysis of the trajectory (video description, instruction–video match, task success). Because temporal-distance labels are derived directly from timestamps, RynnValue scales to **7,000+ hours** of heterogeneous embodied data (~**3M** instruction-conditioned clips) *without any preference or progress annotations*. The predicted time-to-completion is a dense, task-grounded value signal that can be used directly as a progress estimator, a reward model for policy evaluation and ranking, or a critic for reinforcement learning of vision-language-action (VLA) policies.

<p align="center">
  <img src="assets/figures/RynnValue_arch.png" width="100%" alt="RynnValue overview"/>
</p>

> **Overview.** Given a language instruction and a sequence of sampled observations, RynnValue builds an interleaved multimodal sequence of repeated absolute-value (`<value>`) and relative-value (`<relative_value>`) query groups, encoded by the RynnBrain backbone in a single forward pass. Two distributional heads predict the absolute temporal distance to task completion and the signed relative temporal displacement between observations, while the LM head produces video analysis and task verification. The resulting temporal values serve as a unified interface for progress estimation, failure detection, and reward specification in robotic RL.

This repository bundles the complete stack:

| Component | Directory | What it does |
|---|---|---|
| **RynnValue model** | [`rynn_value/`](rynn_value/) | HuggingFace-compatible model definition (config / model / processor), value heads, value tokenizer, custom attention |
| **Inference demo** | [`rynn_infer/`](rynn_infer/) | Run RynnValue on any video + instruction, render an annotated video with the live remaining-time curve |
| **Reward-model benchmark** | [`robometer/`](robometer/) | Fork of [Robometer](https://github.com/robometer/robometer-policy-learning) with a `rynnvalue` baseline adapter: reward alignment, policy ranking, and confusion-matrix evaluation, plus an HTTP reward server |
| **VLA policy RL** | [`pi-rl/`](pi-rl/) | Fork of [openpi](https://github.com/Physical-Intelligence/openpi) (π₀ / π₀-FAST / π₀.₅) extended with offline **IQL** fine-tuning and online **DSRL-style SAC** latent steering, on LIBERO, RoboTwin, and real Franka robots |
| **Tools** | [`tools/`](tools/) | Convert training checkpoints to standalone HuggingFace `trust_remote_code` models |
| **Example** | [`example/`](example/) | Demo video for the inference script |

---

## Table of Contents

- [Highlights](#highlights)
- [How RynnValue Works](#how-rynnvalue-works)
- [Training Recipe](#training-recipe)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart: Inference](#quickstart-inference)
- [Using RynnValue Programmatically](#using-rynnvalue-programmatically)
- [Evaluation with Robometer](#evaluation-with-robometer)
- [Reward Server](#reward-server)
- [Policy RL with pi-rl](#policy-rl-with-pi-rl)
- [Checkpoint Conversion](#checkpoint-conversion)
- [Model Zoo](#model-zoo)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Highlights

- **Temporal distance as the scaling target.** Instead of preferences or normalized `[0,1]` progress, RynnValue predicts the *goal-conditioned cost-to-go in physical seconds*. Labels come directly from timestamps (plus subtask segmentation and cutoff relabeling), so supervision scales to 7,000+ hours / ~3M clips across 10 heterogeneous data sources (AgiBot, EgoDex, Open X-Embodiment, RoboMIND, RoboTwin, …) without a single preference or progress annotation.
- **State-of-the-art without preference labels.** RynnValue-8B attains an average **Kendall's τₐ of 0.675 on RBM-EVAL-OOD**, surpassing the fully preference-supervised state of the art (0.655) and more than doubling a progress-only counterpart (0.292), while generalizing zero-shot to unseen tasks, embodiments, and viewpoints.
- **Shortcut-suppression by design.** *Random temporal sampling* and *temporal-order shuffling* break the correspondence between sequence position / sampling interval and task progress; *value-isolation attention* (`pred_slot_isolated_eager`) keeps each value-query group visible only to its own language–visual context, so predictions can't extrapolate from other value tokens. Ablations: removing shuffling drops τₐ from 0.675 → 0.189, removing isolation → 0.482, uniform sampling → 0.379.
- **Distributional value heads.** Absolute (`[0, 512]` s) and relative (`[−256, 256]` s) temporal targets are discretized into 256 symlog-spaced bins with **two-hot** encodings, decoded back to continuous seconds at inference — stable regression over the long-tailed duration distribution of multi-embodiment data.
- **Language-grounded analysis.** The model also generates an `Analysis` block: a video description, a *Match: Yes/No* verdict (does the video match the instruction?), and a *Success: Yes/No* verdict. Instruction-mismatch augmentation (10% of training samples) teaches the model to detect instruction–video mismatches instead of always reporting smooth progress.
- **A practical reward interface for real robots.** Converted into dense rewards via potential-based shaping (`Φ_t = −v_t`), RynnValue raises real-world dual-arm Franka policy success from **52.5% → 72.5% online** and **63.8% → 82.5% offline** over the strongest reward-model baseline.
- **Full evaluation + RL loop.** The bundled Robometer fork benchmarks RynnValue against 10+ reward-model baselines (RBM, GVL, ReWiND, RFM, RL-VLM-F, Robo-Dopamine, RoboReward, TopReward, VLAC, …); the pi-rl fork closes the loop by improving π₀.₅ policies with value-driven offline IQL and online SAC.

## How RynnValue Works

```
                ┌────────────────────────────────────────────────────────┐
Instruction ───►│                                                        │
Metadata  ─────►│   RynnBrain backbone (RynnValueLangModel)              │──► Analysis text
Frames  ───────►│                                                        │    (description / Match / Success)
                │  hidden states at repeated query-token positions       │
                └───────┬──────────────────────────────┬─────────────────┘
                        │ <value> group (×N)           │ <relative_value> group (×N)
                        ▼                              ▼
                absolute value head            relative value head
              (256 symlog bins, two-hot)     (256 symlog bins, two-hot)
                        │                              │
                        ▼                              ▼
              remaining time per frame     signed Δt between adjacent frames
```

The architecture follows the paper (§2, Model Architecture):

- **Grouped temporal queries.** A single query token is an information bottleneck; each temporal prediction instead uses a group of *N* = 8 repeated query tokens whose hidden states are concatenated (not averaged) before the head, preserving complementary visual cues (object configuration, robot–object interaction, task stage, completion evidence).
- **Dual distributional heads.** The absolute head predicts the remaining time to the (relabeled) completion cutoff; the relative head predicts the signed temporal displacement between consecutively presented observations. Both are 256-bin symlog-spaced two-hot classifiers, decoded at inference by taking the expected bin center in symlog space and applying `symexp`.
- **Value-isolation attention.** Query groups belonging to different observations cannot attend to each other, and context tokens cannot attend to query tokens — each temporal estimate must be grounded in the instruction and visual evidence, and value prediction never contaminates language generation.
- **Language analysis & verification.** A verification prompt after the last query group triggers autoregressive generation of `Video Description → Match → Success` through the original (frozen) LM head.

The prompt (built by `rynn_value/conversations.py`) interleaves an optional embodiment/camera meta block, the task instruction, the two value questions, per-frame images followed by their `<relative_value>` / `<value>` token slots, and finally the analysis request. The processor (`rynn_value/processing_rynn_value_lang.py`) exposes a single-call inference API, `process_episode(instruction, images, robot_description, camera_description)`.

Key implementation files:

| File | Contents |
|---|---|
| `rynn_value/configuration_rynn_value_lang.py` | `RynnValueLangConfig` (extends `Qwen3VLConfig`), `ValueTokenizerConfig` (bins, support transform, encoding), `ValueHeadConfig` (linear / BRO) |
| `rynn_value/modeling_rynn_value_lang.py` | `RynnValueLangModel` — value-head ensemble, relative head, cross-entropy losses over encoded targets, mismatch fusion mask, `RynnValueLangOutputWithPast` |
| `rynn_value/value_tokenizer.py` | Scalar ↔ bin-distribution codec (two-hot encoding) |
| `rynn_value/value_heads.py` | `LinearValueHead`, `BroValueHead` / `BroNet` |
| `rynn_value/attention_impl.py` | `pred_slot_isolated_eager` attention registration |
| `rynn_value/processing_rynn_value_lang.py` | Processor: special tokens, training target construction, `process_episode` inference API |

The package self-registers with HuggingFace Auto classes (`AutoConfig` / `AutoModel` / `AutoProcessor` under model type `rynn_value_lang`), so exported checkpoints load with `trust_remote_code=True` and nothing else.

## Training Recipe

<p align="center">
  <img src="assets/figures/RynnValue-train.png" width="100%" alt="RynnValue training pipeline and value-isolation attention"/>
</p>

> **(a) Training strategy.** Random temporal sampling and temporal-order shuffling suppress shortcuts tied to sampling intervals and sequence position, while instruction-mismatch augmentation strengthens language–visual grounding. **(b) Value-isolation attention.** Within each value-query group, repeated queries attend to one another and to the language–visual context, while remaining isolated from other value-query groups.

**Data.** RynnValue is trained on a heterogeneous mixture of real-world, simulated, and egocentric trajectories — 1.67M original episodes expanded into **3.09M instruction-conditioned segments** (7,000+ hours, 223K unique instructions) via subtask segmentation and cutoff relabeling:

| Data Source | # Original Episodes | # Segmentations |
|---|---:|---:|
| Open X-Embodiment | 693,037 | 693,037 |
| EgoDex | 338,234 | 338,234 |
| InternData-A1 | 320,905 | 320,905 |
| AgiBot | 167,535 | 1,166,042 |
| RoboCOIN | 67,420 | 410,877 |
| RoboMIND | 32,138 | 32,138 |
| RoboTwin | 27,414 | 27,414 |
| Galaxea Open-World | 16,979 | 95,671 |
| RDT | 6,109 | 6,109 |
| Soft-FOLD | 1,542 | 1,542 |
| **Total** | **1,671,313** | **3,091,969** |

Temporal-distance labels are generated directly from timestamps: observations before the completion cutoff are labeled with their remaining time; observations at or after the cutoff receive zero. Qwen3-VL-27B captions supervise the `Video Description` output.

**Objectives.** Three jointly optimized cross-entropy losses: (1) absolute temporal-distance loss over two-hot bin targets (masked for instruction-mismatched samples); (2) relative temporal-distance loss (instruction-independent, kept for mismatched samples); (3) causal LM loss over the `Video Description / Match / Success` tokens (weight λ = 2, LM output projection frozen).

**Shortcut suppression.** For each clip, *K* = 8 observations are sampled at irregular timestamps (random temporal sampling); half of the sequences are unsorted, the rest follow a forward-biased temporal walk with occasional rewinds (temporal-order shuffling) — so relative targets can be negative. Together with value-isolation attention, this forces every prediction to be grounded in the corresponding observation and task semantics. For 10% of samples the instruction is swapped with one from a different trajectory (instruction-mismatch augmentation) and supervised toward `Match: No` / `Success: No`.

**Reward interface.** At inference, frames are fed chronologically and decoded into remaining time `v_t`. The potential `Φ_t = −v_t` yields dense rewards via potential-based shaping (`r_t = γ^H Φ_{t+H} − Φ_t`), preserving the physical temporal scale rather than normalizing to a task-specific `[0,1]` interval.

## Results

**Policy ranking (RBM-EVAL-OOD).** Trained without preference labels, RynnValue-8B reaches an average **Kendall's τₐ of 0.675**, surpassing the fully preference-supervised state of the art (0.655) and more than doubling a progress-only counterpart (0.292).

**Instruction–trajectory alignment.** Scoring every instruction against every trajectory, RynnValue produces the clearest diagonal structure with the highest normalized diagonal margin (**0.79** vs. 0.67 for the strongest baseline):

<p align="center">
  <img src="assets/figures/Matrix.png" width="100%" alt="Instruction-trajectory confusion matrices"/>
</p>

**Value-curve quality.** On real-world trajectories, RynnValue reacts sharply to task regressions and recoveries where normalized-progress baselines stay flat:

<p align="center">
  <img src="assets/figures/value-case.png" width="100%" alt="Temporal-value curve comparison on a real-world trajectory"/>
</p>

**Scaling behavior.** Task diversity — not episode volume — drives generalization: scaling episode count within fixed tasks saturates almost immediately, while adding tasks monotonically reduces temporal-distance error on unseen tasks:

<p align="center">
  <img src="assets/figures/scaling_analysis.png" width="55%" alt="Scaling episode volume vs. task diversity"/>
</p>

**Real-world policy learning.** Used as a zero-shot reward annotator (none of the tasks, objects, or scenes appear in training) on a dual-arm Franka across four manipulation tasks, RynnValue-shaped rewards raise average success from **52.5% → 72.5% (online RL)** and **63.8% → 82.5% (offline RL)** over the strongest reward-model baseline:

<p align="center">
  <img src="assets/figures/case_study.png" width="100%" alt="Representative real-world manipulation tasks"/>
</p>

## Repository Structure

```
RynnValue001/
├── rynn_value/                 # RynnValue model package (HF trust_remote_code style)
├── rynn_infer/
│   ├── inference.py            # CLI: video + instruction → value curve + analysis
│   ├── plot_utils.py           # save_video_with_trend(): renders annotated output video
│   └── outputs/                # sample inference outputs
├── robometer/                  # Robometer benchmark fork (reward-model training + eval)
│   ├── robometer/
│   │   ├── configs/            # Hydra configs (reward_model/rynnvalue.yaml, distributed/fsdp.yaml, …)
│   │   ├── models/             # RBM (progress/preference/success heads), ReWiND transformer
│   │   ├── evals/              # run_baseline_eval.py, eval/baseline servers, baselines/rynnvalue.py
│   │   ├── data/ trainers/     # dataset + training code
│   ├── rynnvalue_eval/         # RynnValue-specific eval launchers (policy ranking, confusion matrix, server)
│   ├── eval_commands/          # eval commands for the other baselines
│   ├── dataset_upload/         # converters to the RBM HF dataset format (LIBERO, AgiBotWorld, custom)
│   └── train.py                # reward-model training entry (accelerate + FSDP, LoRA)
├── pi-rl/                      # openpi fork + RL
│   ├── src/openpi/             # π₀ / π₀.₅ models (JAX + PyTorch), training, RL data loading
│   ├── scripts/                # train.py, train_iql.py, serve_policy.py, launch scripts
│   ├── configs/                # RL configs (SAC, TD, DAgger, EXPO)
│   ├── examples/               # libero, droid, aloha, dsrl_sim, dsrl_franka, franka (real robot), …
│   ├── packages/openpi-client/ # websocket policy client
│   └── third_party/jaxrl2/     # vendored jaxrl2 (pi_iql, pixel_iql, pixel_sac)
├── tools/
│   └── convert_rynn_value_lang_to_hf.py   # training ckpt → standalone HF model
└── example/
    └── Put_the_box_in_the_drawer_and_close_it.mp4
```

## Installation

Each component manages its own environment. Python 3.10 is required throughout.

### RynnValue inference (`rynn_value` + `rynn_infer`)

Managed with [uv](https://docs.astral.sh/uv/) via the top-level `pyproject.toml`:

```bash
uv sync                     # creates .venv with torch / transformers / imageio / …
```

A recent `transformers` with Qwen3-VL support is required (pinned in `pyproject.toml`). A single GPU with ≥ 24 GB memory comfortably runs the 8B model in bf16.

### Robometer evaluation

```bash
cd robometer
uv sync                     # Python 3.10, uv-managed (see pyproject.toml / uv.lock)
```

See `robometer/README.md` and `robometer/FINETUNE_ROBOMETER.md` for dataset download, reward-model fine-tuning (LoRA + FSDP), and the full baseline matrix.

### pi-rl

```bash
cd pi-rl
GIT_LFS_SKIP_SMUDGE=1 uv sync           # JAX 0.5.3 (cuda12), flax, torch 2.7.1, lerobot, …
```

GPU requirements follow upstream openpi: > 8 GB for inference, > 22.5 GB for LoRA fine-tuning, > 70 GB (A100/H100) for full fine-tuning. See `pi-rl/README.md`.

## Quickstart: Inference

Run RynnValue on the bundled example video:

```bash
cd rynn_infer
uv run python inference.py \
    --model_path /path/to/RynnValue-8B \
    --video_path ../example/Put_the_box_in_the_drawer_and_close_it.mp4 \
    --instruction "Put the box in the drawer and close it" \
    --num_frames 64 \
    --output_path ./outputs
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--num_frames` | 64 | Frames uniformly sampled from the video (0 = all frames) |
| `--robot_description` / `--camera_description` | None | Embodiment/camera meta block (required for models trained with `use_meta=True`) |
| `--max_new_tokens` | 128 | Token budget for the Analysis block |
| `--fps` | 30 | FPS of the rendered trend video |

The script produces, in a timestamped output directory:

- `output_with_trend.mp4` — the input video with a synchronized **Remaining Time (s)** curve;
- the parsed Analysis (video description, `Match: Yes/No`, `Success: Yes/No`) and per-frame values.

## Using RynnValue Programmatically

```python
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

model_path = "/path/to/RynnValue-8B"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config._attn_implementation = "pred_slot_isolated_eager"

model = AutoModel.from_pretrained(
    model_path, config=config, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="cuda",
).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# `images`: list of PIL.Image frames sampled from the trajectory video
inputs = processor.process_episode(
    instruction="Put the box in the drawer and close it",
    images=images,
).to(model.device)

with torch.no_grad():
    out = model(**inputs)

remaining_time = out.value.pred_value.float().mean(dim=-1)   # (num_frames,) seconds, head-ensemble mean
delta_time     = out.relative.pred_value                     # per-step time deltas
entropy        = out.value.entropy                           # per-frame uncertainty
```

## Evaluation with Robometer

The `robometer/` fork adds RynnValue as a first-class baseline (`robometer/robometer/evals/baselines/rynnvalue.py`, Hydra config `reward_model=rynnvalue`). Ready-made example commands live in `robometer/rynnvalue_eval/` (plus `start_server.sh` for the reward server).

**Policy ranking** (does the value model rank better policies higher?) — example commands in `rynnvalue_eval/policy_ranking.sh`; the released-model run:

```bash
cd robometer
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=Alibaba-DAMO-Academy/RynnValue-8B' \
    'custom_eval.eval_types=[policy_ranking]' \
    'custom_eval.policy_ranking=[rbm-1m-ood]' \
    'custom_eval.use_frame_steps=false' \
    'custom_eval.pad_frames=false' \
    'custom_eval.num_examples_per_quality_pr=1000' \
    'max_frames=8' \
    'model_config.checkpoint_path=null' \
    'model_config.mode=absolute' \
    'model_config.stride=2' \
    'model_config.num_frames=8' \
    'model_config.camera_desc_lookup_path=./extracted_meta_with_descriptions.json'
```

Two optional `model_config` fields: `camera_desc_lookup_path=extracted_meta_with_descriptions.json` supplies the per-trajectory camera descriptions used by RBM-EVAL-OOD, and `attn_implementation` (`eager` / `sdpa`) overrides the attention implementation, defaulting to the model's own `pred_slot_isolated_eager` value-isolation attention. To evaluate a fine-tuned raw checkpoint instead, set `model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX` (a `model.pt` directory whose sibling `huggingface/` holds the processor/config artifacts).

**Confusion matrix** (cross-task instruction/video matching) — example commands in `rynnvalue_eval/confusion_matrix.sh`, one per scoring mode (`model_config.confusion_score_mode`):

- `match_binary` — score 1.0 iff the Analysis verdict is `Match: Yes`;
- `normalized_value` — the value head normalized to `[0, 1]` (`1 − t/t_max`) on matches, 0 otherwise.

```bash
cd robometer
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=Alibaba-DAMO-Academy/RynnValue-8B' \
    'custom_eval.eval_types=[confusion_matrix]' \
    'custom_eval.confusion_matrix=[[...dataset ids...]]' \
    'max_frames=8' \
    'model_config.stride=2' \
    'model_config.num_frames=8' \
    'model_config.camera_desc_lookup_path=./extracted_meta_with_descriptions.json' \
    'model_config.confusion_score_mode=match_binary'
```

See `robometer/eval_commands/` for the other baselines. Dataset converters for LIBERO, AgiBotWorld, and custom datasets (DROID / Bridge style) live in `robometer/dataset_upload/`.

### Policy Ranking Reproduction Results

Reproduced results of the released RynnValue-8B checkpoint on RBM-EVAL-OOD policy ranking (`max_frames=8`, `stride=2`, `num_frames=8`, per-trajectory camera descriptions):

| Dataset | `pred_slot_isolated_eager` | `eager` | `sdpa` |
|---|---|---|---|
| usc_koch_p_ranking_all (rfm) | 0.483 | **0.555** | 0.539 |
| rfm_new_mit_franka | 0.468 | **0.510** | 0.492 |
| usc_franka | 0.625 | **0.667** | **0.667** |
| usc_trossen | 1.000 | 1.000 | 1.000 |
| usc_xarm | 0.472 | 0.472 | **0.500** |
| utd_so101_clean_top | 0.833 | 0.833 | 0.833 |
| **Average** | 0.647 | **0.673** | 0.672 |

Kendall's τ (last-frame), reproduced via `rynnvalue_eval/policy_ranking.sh`; the columns correspond to `model_config.attn_implementation` unset (model default `pred_slot_isolated_eager`), `=eager`, and `=sdpa`.

## Reward Server

Serve RynnValue as an HTTP reward model (used by policy evaluation and online RL clients):

```bash
cd robometer
bash rynnvalue_eval/start_server.sh \
    --model-path /path/to/RynnValue-8B \
    --port 8001 --gpu 0 --num-frames 8 --batch-size 16
```

The server (`robometer/robometer/evals/baseline_eval_server.py`) accepts frame sequences + instructions and returns per-frame values / rewards. `--checkpoint-path` alternatively loads a raw training checkpoint (`model.pt` with a sibling `huggingface/` snapshot). `--mode absolute` selects the remaining-time head; `--debug` attaches `debugpy` on `:5678`.

## Policy RL with pi-rl

`pi-rl/` is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) that turns value/reward signals into better VLA policies. On top of upstream π₀ / π₀-FAST / π₀.₅ SFT (JAX and PyTorch), it adds:

- **Offline IQL fine-tuning** (`scripts/train_iql.py`): each step (1) updates a jaxrl2 `PixelIQL` critic/value, (2) computes the IQL advantage, (3) updates the π₀.₅ flow-matching policy with advantage-weighted BC (`exp(A_scaling · adv)`). RL-aware plumbing lives in `src/openpi/training/{rl_data_loader,iql_checkpoints,episode_filter}.py`. Registered configs include `pi05_robotwin_iql`, `pi05_franka_single_iql`, `pi05_franka_dual_iql`, and variants.
- **Online DSRL-style SAC latent steering**: a small `PixelSAC` agent steers the frozen π₀.₅ policy's latent noise online — in simulation (`examples/dsrl_sim/`, LIBERO `OffScreenRenderEnv`) and on a real Franka over WebSocket (`examples/dsrl_franka/`).
- **Benchmarks & robots**: LIBERO, RoboTwin, ALOHA sim, DROID, and a full real-Franka pipeline (`examples/franka/`: serving, fine-tuning, async online LoRA training) with LeRobot-format data converters (`scripts/convert_franka_data_to_lerobot.py`).

Representative commands:

```bash
cd pi-rl

# supervised fine-tuning on RoboTwin
FSDP_DEVICES=2 bash scripts/train_pi05_robotwin.sh adjust_bottle-demo_clean_collect_200-50

# offline IQL on RoboTwin
bash scripts/train_iql_robotwin.sh
```

### DSRL Online RL on a Real Franka

`examples/dsrl_franka/` runs online SAC latent steering on a real (single- or dual-arm) Franka, optionally shaped by a RynnValue reward server. The launcher template is `examples/dsrl_franka/scripts/run_train_franka.sh`; the flow is:

**1. (Optional) Start the RynnValue reward server** for reward shaping (see [Reward Server](#reward-server)):

```bash
cd robometer && bash rynnvalue_eval/start_server.sh --model-path /path/to/RynnValue-8B --port 8000
```

**2. Start the robot environment server** (WebSocket) on the machine controlling the Franka, or skip this and use `--fake_env` for a quick dry run without hardware.

**3. Launch online training.** Point it at an IQL-initialized checkpoint (`FRANKA_SFT_CKPT_BASE`) and the env/reward servers:

```bash
cd pi-rl
export FRANKA_SFT_CKPT_BASE=/path/to/iql_checkpoint/10000   # offline-IQL warm start
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python examples/dsrl_franka/launch_train_franka.py \
    --arm_mode dual \
    --policy_config pi05_franka_dual_iql_optimized_v2 \
    --update_type episode --utd_ratio 100 --publish_hz 10.0 \
    --client_host localhost --client_port 8101 \
    --pi0_action_horizon 16 \
    --franka_norm_stats_asset_id pick_up_the_box \
    --max_episodes 500 --franka_max_timesteps 600 \
    --start_online_updates 200 \
    --noise_episodes 2 --noise_std 0.1 \
    --score_server "http://localhost:8000" \
    --shaping_weight 1.0 --shaping_gamma 0.999 \
    --task_description "Move the box from the right side to the left side." \
    --checkpoint_interval 200 --checkpoint_dir ./experiments/dual_reward_shaping \
    --wandb_project dsrl_franka --seed 42
```

Key knobs:

| Flag | Meaning |
|---|---|
| `--arm_mode single/dual` (+ `--side left/right`) | Single- or dual-arm Franka |
| `--fake_env` | Local fake environment — verify the training loop without a robot |
| `--client_host` / `--client_port` | WebSocket address of the robot env server |
| `--score_server` | RynnValue/Robometer reward server URL; enables potential-based reward shaping (`--shaping_weight`, `--shaping_gamma`) |
| `--update_type episode` / `--utd_ratio` | Update after each episode with the given update-to-data ratio |
| `--noise_episodes` / `--noise_std` | Initial exploration episodes with Gaussian latent noise |
| `--start_online_updates` | Warm-up steps collected before SAC updates begin |

The same recipe runs in simulation via `examples/dsrl_sim/launch_train_sim.py` (LIBERO `OffScreenRenderEnv`).

See `pi-rl/README.md` for upstream documentation (checkpoints under `gs://openpi-assets/checkpoints/`: `pi05_base`, `pi05_libero`, `pi05_droid`, …), norm-stat computation, and policy serving.

## Checkpoint Conversion

Export a raw training checkpoint to a standalone HuggingFace model directory (embeds the modeling code for `trust_remote_code` loading):

```bash
uv run python tools/convert_rynn_value_lang_to_hf.py \
    --model_ckpt_path /path/to/checkpoint_model_XXXXXX \
    --output_path /path/to/RynnValue-8B-hf \
    --dtype bf16
```

`--model_ckpt_path` expects a directory containing `model.pt` with the matching `huggingface/` processor/config snapshot next to it.

## Model Zoo

| Model | Backbone | Description |
|---|---|---|
| [RynnValue-4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-4B) | [RynnBrain-4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-4B) | Remaining-time value model with absolute + relative distributional heads and Analysis generation |
| [RynnValue-8B](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B) | [RynnBrain-8B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B) | Remaining-time value model with absolute + relative distributional heads and Analysis generation |

Checkpoint release is in progress; update the paths above once the weights are published.

## Acknowledgements

This repository is built on top of the following open-source works:

- [**openpi**](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence) — π₀ / π₀-FAST / π₀.₅ VLA models and training stack (Apache-2.0; `pi-rl/`).
- [**Robometer**](https://github.com/robometer/robometer-policy-learning) — "Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons": benchmark, RBM baselines, and the RBM-1M dataset (`robometer/`).
- [**jaxrl2**](https://github.com/ikostrikov/jaxrl2) — IQL / SAC agents (vendored in `pi-rl/third_party/jaxrl2/`).
- [**RynnBrain**](https://github.com/alibaba-damo-academy/RynnBrain) - Open Embodied Foundation Models
- [**Qwen3-VL**](https://github.com/QwenLM/Qwen3-VL) — the underlying vision-language architecture of the RynnBrain backbone.
- Simulation benchmarks: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [RoboTwin](https://github.com/TianxingChen/RoboTwin), [DROID](https://droid-dataset.github.io/), gym-aloha.

## License

- The RynnValue components (`rynn_value/`, `rynn_infer/`, `tools/`) are distributed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
- `pi-rl/` is distributed under the **Apache License 2.0** (see `pi-rl/LICENSE`) with additional Gemma terms in `pi-rl/LICENSE_GEMMA.txt`.
- `robometer/` follows the upstream Robometer project under the **MIT License** (see `robometer/LICENSE`). It additionally vendors FSDP utilities derived from ByteDance's [verl](https://github.com/volcengine/verl) project (Apache-2.0; original copyright headers retained in `robometer/robometer/utils/fsdp/`).

## Citation

If you find RynnValue useful, please cite:

```bibtex
@article{rynnvalue2026,
  title  = {RynnValue: Scaling Robotic Value Foundation Models with Temporal Distance},
  author = {Dongchi Huang and Hongyin Zhang and Bohan Hou and Siteng Huang and Zhian Su and Hang Guo and Tong Lu and Zhaofeng Xu and Jiahao Tang and Jianfei Yang and Donglin Wang and Peixi Peng and Mingxiu Chen and Deli Zhao and Xin Li},
  journal= {arXiv preprint arXiv:2608.09853},
  year   = {2026},
}
```

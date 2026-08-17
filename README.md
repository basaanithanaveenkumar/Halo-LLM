<div align="center">

# ha-llm

Compare **autoregressive**, **masked diffusion**, **block diffusion**, and **flow matching** on one transformer backbone.

</div>

<p align="center">
  <img src="assets/block_diffusion_step4900.gif" alt="Block diffusion decoding on WikiText — tokens unmask block by block" width="100%">
</p>

<p align="center">
  <em>Block diffusion at step 4900: prompt in steelblue, committed tokens in orange, the active block in black on a light overlay.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/pytorch-2.13+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/tokenizer-GPT--2-yellow" alt="GPT-2 tokenizer">
  <img src="https://img.shields.io/badge/package-ha__llm-0A7A3E" alt="ha_llm">
</p>

---

## Why this repo

The same `TransformerBackbone` is reused for every paradigm. Masking, attention mechanism, and FFN are config switches — not forked model files. Train, sample, eval, and visualize through one CLI.

| Switch | Options |
|---|---|
| Variant | `autoregressive` · `diffusion` · `block_diffusion` · `flow_matching` · `mtp` (stub) |
| Attention mask | `causal` · `bidirectional` · `block_causal` |
| Attention impl | `mha` · `gqa` · `mqa` |
| FFN | `mlp` · DeepSeek-style `moe` |
| Token loss | `ce` · `focal` |

Shared core (`core/`, `dataloader/`, `training/`, `inference/`, `metrics/`) never imports a `models_*` package by name. Each paradigm lives in `src/ha_llm/models/models_<name>/` and registers with `@register_model` / `@register_loss` / `@register_sampler` / `@register_metric`.

---

## Quick start

```bash
uv sync
uv sync --extra dev
uv run pytest tests/unit -q
```

Train block diffusion on full WikiText-2 (MPS / CUDA via `device` in YAML):

```bash
uv run ha-llm-train --config configs/block_diffusion/small.yaml --no-resume --viz
```

Sample, eval, and write a GIF from `latest` or a named run:

```bash
uv run ha-llm-sample --config configs/block_diffusion/small.yaml
uv run ha-llm-eval --config configs/block_diffusion/small.yaml
uv run ha-llm-viz --config configs/block_diffusion/small.yaml
tensorboard --logdir data/experiments
```

---

## Variants

| Config | What it does |
|---|---|
| [`configs/autoregressive/small.yaml`](configs/autoregressive/small.yaml) | Next-token LM (reference) |
| [`configs/diffusion/small.yaml`](configs/diffusion/small.yaml) | Absorbing-state masked diffusion |
| [`configs/block_diffusion/small.yaml`](configs/block_diffusion/small.yaml) | AR across blocks, diffusion inside the block |
| [`configs/flow_matching/small.yaml`](configs/flow_matching/small.yaml) | Discrete flow matching |
| [`configs/mtp/small.yaml`](configs/mtp/small.yaml) | Multi-token heads (stub) |

Default small model: `d_model=384`, 16 layers, 8 heads, `d_ff=1536`, `max_length=128`, GPT-2 tokenizer, WikiText-2 with a **10-word sliding stride**.

---

## Experiments

Each run is namespaced:

```
data/experiments/<variant>/<variant>_<YYYYMMDD_HHMMSS>/
  config.yaml          # resolved settings
  configs/             # source YAML chain
  checkpoints/last.pt
  logs/train.log
  tb/                  # TensorBoard
  viz/                 # generation GIFs
```

`latest` is a symlink to the newest folder. Resume with `--resume` (uses `latest`) or `--experiment NAME`.

Training GIFs write every `viz.every_n_steps` (default 50) using a short sampler. Prompts are taken from WikiText validation (`viz.prompt_source: dataset`). Override with `--prompt`.

---

## Config knobs

Set these in [`configs/base.yaml`](configs/base.yaml) or a variant YAML:

```yaml
model:
  attn_impl: mha          # mha | gqa | mqa
  ffn_type: mlp           # mlp | moe
train:
  loss_type: ce           # ce | focal
  focal_gamma: 2.0
data:
  stride_words: 10
viz:
  every_n_steps: 50
  sampling_steps: 8
  prompt_source: dataset
device: mps               # or cuda / cpu
```

---

## Adding a paradigm

1. New folder `src/ha_llm/models/models_<name>/` with model, loss, sampler (and optional metric).
2. One import in `src/ha_llm/models/__init__.py`.
3. One YAML under `configs/<name>/`.

Public facades stay the same: `train()`, `evaluate()`, `load_session()`.

---

## Tests

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration/test_overfit_sanity.py -k autoregressive -v
```

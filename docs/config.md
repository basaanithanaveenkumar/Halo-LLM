# Configuration reference

Every run is a YAML file under `configs/`. Variant files **inherit** [`configs/base.yaml`](../configs/base.yaml) and override only what they need. Unknown keys are rejected (`extra: forbid` in the Pydantic schema).

```yaml
inherits: ../base.yaml
variant: block_diffusion
model:
  arch: lgt
```

Schema lives in [`src/ha_llm/config/sections/`](../src/ha_llm/config/sections/) (one file per YAML block). [`RunConfig`](../src/ha_llm/config/run.py) composes them; [`load_config`](../src/ha_llm/config/load.py) handles YAML inheritance.

Pass a file to the CLI:

```bash
uv run ha-llm-train --config configs/block_diffusion/small.yaml --no-resume --viz
```

CLI flags (`--epochs`, `--steps`, `--resume`, `--prompt`, `--experiment`) override YAML after load.

---

## Inheritance

1. Load the file you pass (`--config`).
2. If it has `inherits:`, load that parent (and its parent, one extra hop).
3. Deep-merge: nested dicts merge; scalars in the child **replace** the parent.

`inherits` is not a `RunConfig` field; it is stripped during merge.

Put shared size, data, and logging in `base.yaml`. Put paradigm-specific `variant`, `model.attn_type`, `model.arch`, and `model.block_size` in `configs/<variant>/small.yaml`.

---

## Top-level

| Key | Type | Default | What it does |
|---|---|---|---|
| `variant` | string, **required** | — | Which registered model/loss/sampler to use: `autoregressive`, `diffusion`, `block_diffusion`, `flow_matching`, `mtp`. Must match a `@register_variant` name. |
| `device` | `mps` \| `cuda` \| `cpu` \| `null` | `null` (auto) | Training device. `null` lets the runtime pick. Example: `device: mps` on Apple Silicon. |
| `model` | object | see below | Architecture. |
| `train` | object | see below | Optimizer loop. |
| `data` | object | see below | Dataset and tokenizer. |
| `sample` | object | see below | Generation. |
| `eval` | object | see below | Validation metrics. |
| `viz` | object | see below | Unmasking GIFs. |
| `logging` | object | see below | Loguru + TensorBoard. |
| `experiment` | object | see below | Run folders under `data/experiments/`. |

---

## `model`

Shared width/depth for every variant. `arch` chooses the **block stack**; `attn_type` chooses the **mask**.

### Size

| Key | Type | Default (schema / `base.yaml`) | How to use |
|---|---|---|---|
| `d_model` | int | `128` / `384` | Residual stream width. Must be divisible by `n_heads`. Larger → more capacity and VRAM. |
| `n_heads` | int | `4` / `8` | Query heads. |
| `n_layers` | int | `4` / `16` | Transformer / DiT / LGT depth. |
| `d_ff` | int | `512` / `1536` | FFN inner size (per expert if MoE). Typical ~4× `d_model`. |
| `dropout` | float | `0` / `0.1` | Attention and FFN dropout. |
| `max_length` | int | `64` / `128` | Tokens per training window and position table. Sequence length must be divisible by `block_size` for block diffusion. |

### Backbone (`arch`)

| Value | Stack | Typical use |
|---|---|---|
| `transformer` or `default` | GPT-style pre-norm + LayerNorm + learned positions | Autoregressive, MTP |
| `lgt` | Local-global transformer: RMSNorm sandwich, GQA, dual RoPE, 5:1 sliding-window / global, GeGLU | Diffusion, block diffusion, flow matching |
| `dit` | AdaLN-Zero DiT blocks (scale, shift, residual gates) | Same denoisers, alternative to LGT |

```yaml
model:
  arch: lgt    # or dit, or transformer
```

Swap without changing variant code: all models call `build_backbone(...)`.

### Attention mask (`attn_type`)

| Value | Meaning | Variant |
|---|---|---|
| `causal` | Token *i* attends to *≤ i* | `autoregressive`, `mtp` |
| `bidirectional` | Full self-attention | `diffusion`, `flow_matching` |
| `block_causal` | AR across blocks, bidirectional inside the current block (clean + noised streams) | `block_diffusion` |

### Attention implementation

| Key | Type | Default | How to use |
|---|---|---|---|
| `attn_impl` | `mha` \| `gqa` \| `mqa` | `mha` | `mha` = PyTorch `MultiheadAttention`. `gqa` = grouped-query (set `n_kv_heads`). `mqa` = one KV head. LGT forces GQA internally. |
| `n_kv_heads` | int \| `null` | `null` | KV heads for GQA. Must divide `n_heads`. `null` → `n_heads/4` for GQA, or LGT local `n_heads/2` and global `n_heads/8`. Example: `n_heads: 8`, `n_kv_heads: 2`. |

### FFN

| Key | Type | Default | How to use |
|---|---|---|---|
| `ffn_type` | `mlp` \| `geglu` \| `moe` | `mlp` | `mlp` = Linear–GELU–Linear. `geglu` = gated GELU (LGT default if you leave `mlp` on `arch: lgt`). `moe` = noisy top-k experts + shared experts. |
| `moe_num_experts` | int | `4` | Routed experts (ignored unless `ffn_type: moe`). |
| `moe_top_k` | int | `2` | Experts active per token. Must be `≤ moe_num_experts`. |
| `moe_num_shared` | int | `1` | Experts applied to every token. |

VRAM: MoE multiplies FFN parameters. Prefer `geglu` on a laptop; use `moe` when you want sparsity.

### Time, blocks, MTP

| Key | Type | Default | How to use |
|---|---|---|---|
| `use_time_cond` | bool | `false` | AdaLN (transformer/LGT) or DiT timestep embedding. Set `true` for diffusion / block diffusion / flow matching. Leave `false` for AR. |
| `block_size` | int \| `null` | `null` | Tokens per block for `attn_type: block_causal`. Required for block diffusion. `max_length` must be a multiple (e.g. `128` and `16`). |
| `n_mtp_heads` | int | `2` | Extra next-token heads for the MTP stub. Unused by other variants. |

### LGT-only (ignored by `transformer` / `dit`)

| Key | Type | Default | How to use |
|---|---|---|---|
| `sliding_window` | int | `512` | Local layers attend at most this many tokens away. If `≥ max_length`, local ≈ full attention; global layers still use p-RoPE. |
| `local_global_ratio` | int | `5` | Five local layers then one global (`(i+1) % 6 == 0`). |
| `rope_theta_local` | float | `10000` | RoPE base on local layers. |
| `rope_theta_global` | float | `1e6` | RoPE base on global layers. |
| `p_rope` | float | `0.25` | Fraction of head dim rotated on **global** layers (p-RoPE). Local layers rotate the full head. |
| `qk_norm` | bool | `false` | RMSNorm on Q and K. LGT always applies QK-Norm. |

---

## `train`

| Key | Type | Default (schema / `base.yaml`) | How to use |
|---|---|---|---|
| `epochs` | int \| `null` | `null` / `2000` | Full passes over the loader. If set, **wins** over `steps`. CLI: `--epochs`. |
| `steps` | int \| `null` | `300` / `null` | Optimizer steps if `epochs` is `null`. CLI: `--steps` (ignored when `--epochs` is set). |
| `lr` | float | `1e-3` / `3e-4` | AdamW learning rate. |
| `batch_size` | int | `8` / `32` | Sequences per step. `drop_last=True`; if the split is smaller than this, the loader is empty. Lower on MPS (block diffusion uses `28`). |
| `weight_decay` | float | `0` / `0.01` | AdamW decay. |
| `log_every` | int | `50` / `100` | Log loss / lr / grad every N steps. |
| `grad_clip` | float | `1.0` | Global grad-norm clip. `0` or very large disables in practice if you change code; keep `1.0`. |
| `seed` | int | `0` | `torch.manual_seed`. |
| `checkpoint_path` | string | `checkpoints/last.pt` | Overwritten by the experiment layout to `.../checkpoints/last.pt`. |
| `checkpoint_every_epoch` | bool | `true` | Save `last.pt` at epoch end. |
| `resume` | bool | `true` / `false` | Load `last.pt` if present. CLI `--resume` / `--no-resume`. `--no-resume` starts a new timestamped run. |
| `loss_type` | `ce` \| `focal` \| `label_smoothing` | `ce` | Registered token loss. Add more with `@register_token_loss`. |
| `focal_gamma` | float | `2.0` | Focal focusing (`loss_type: focal`). |
| `focal_alpha` | float | `1.0` | Focal class weight. |
| `label_smoothing` | float | `0.1` | Smoothing when `loss_type: label_smoothing`. |

**Schedule:** `epochs: 2000` and `steps: null` → 2000 epochs. `epochs: null` and `steps: 1000` → enough epochs to cover 1000 steps.

---

## `data`

`null` on sizes means **use the entire Hugging Face split**, not “no data.”

| Key | Type | Default | How to use |
|---|---|---|---|
| `source` | `huggingface` \| `overfit` | `huggingface` | Hub corpus vs repeated overfit string. `overfit_text` set also enables overfit. |
| `dataset` | string | `Salesforce/wikitext` | Hub id: `load_dataset(dataset, subset, split=...)`. |
| `subset` | string \| `null` | `wikitext-2-raw-v1` | Config name. WikiText-103: `wikitext-103-raw-v1`. `null` if the dataset has no configs. |
| `train_split` | string | `train` | Split name on the Hub. |
| `val_split` | string | `validation` | Some datasets use `test` or `valid`. |
| `text_field` | string | `text` | Column with raw text. |
| `cache_dir` | string \| `null` | `null` | Hugging Face cache. `null` → default `~/.cache/huggingface`. |
| `tokenizer_name` | string | `gpt2` | `AutoTokenizer.from_pretrained`. Must match a Hub tokenizer. |
| `train_size` | int \| `null` | `null` | Keep the first N Hub **rows** (not windows). `null` = all. Example: `2048` for a smoke run. |
| `val_size` | int \| `null` | `null` | Same for validation. |
| `stride_words` | int | `10` | Sliding window step in **words**. `1` = dense overlap (more windows, slower). Larger = fewer windows. |
| `overfit_text` | string \| `null` | `null` | If set (or `source: overfit`), skip the Hub and pack this string. |
| `n_overfit_copies` | int | `64` | Repeated copies so `batch_size` and `drop_last` still work. |
| `add_special_tokens` | bool | `false` | Reserved for tokenizer encode; sliding windows currently encode without specials. |

### Recipes

Full WikiText-2 (current default):

```yaml
data:
  source: huggingface
  dataset: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  train_size: null
  val_size: null
  stride_words: 10
```

Subset for debugging:

```yaml
data:
  train_size: 512
  val_size: 128
  stride_words: 20
```

WikiText-103:

```yaml
data:
  subset: wikitext-103-raw-v1
```

Overfit:

```yaml
data:
  source: overfit
  overfit_text: "hello world this is a tiny overfit sentence"
  n_overfit_copies: 64
```

---

## `sample`

Used by `ha-llm-sample`, `ha-llm-viz`, and training GIFs (viz can shorten `sampling_steps`).

| Key | Type | Default (schema / `base.yaml`) | How to use |
|---|---|---|---|
| `max_new_tokens` | int | `32` / `64` | Tokens generated after the prompt. |
| `temperature` | float | `1.0` | Softmax temperature. Lower → sharper. |
| `sampling_steps` | int | `32` | Denoising / flow steps. Ignored by greedy AR decode except as unused. More steps → slower, usually cleaner diffusion. |
| `eps` | float | `1e-3` / `0.001` | Minimum noise level (avoid `t=0` in diffusion/flow). |
| `prompt` | string | `"hello"` / `"The meaning of life is"` | Default prompt. CLI `--prompt` overrides. Viz uses this only if `viz.prompt_source: config`. |

---

## `eval`

| Key | Type | Default | How to use |
|---|---|---|---|
| `metrics` | list of strings | `loss`, `perplexity`, `bits_per_token`, `token_accuracy`, `masked_accuracy` | Names from the metric registry. LLM metrics live in `ha_llm.metrics.llm` (`topk_accuracy` is optional). Unknown names error. CLI `ha-llm-eval --metrics loss,perplexity`. |
| `every_n_epochs` | int \| `null` | `1` / `10` | Run eval during training. `null` skips periodic eval. Block diffusion `small.yaml` uses `1`. |
| `max_batches` | int \| `null` | `null` | Cap val batches. `null` = full val loader. Set e.g. `20` to speed training. |

---

## `viz`

Training GIFs go to the experiment `viz/` folder (layout overrides `output_dir`).

| Key | Type | Default (schema / `base.yaml`) | How to use |
|---|---|---|---|
| `enabled` | bool | `false` / `true` | Master switch. CLI `--viz` / `--no-viz`. |
| `every_n_steps` | int \| `null` | `500` / `50` | GIF every N optimizer steps. `null` disables step-based GIFs. Cheap: `500`. Frequent: `50`. |
| `every_n_epochs` | int \| `null` | `null` | Extra GIF at epoch boundaries. |
| `sampling_steps` | int \| `null` | `8` | Denoise steps **inside training GIFs** (keep small). `ha-llm-viz` uses `sample.sampling_steps`. |
| `max_new_tokens` | int \| `null` | `null` | Override generation length for GIFs. `null` → `sample.max_new_tokens`. |
| `prompt_source` | `dataset` \| `config` | `dataset` | `dataset` = prefix from validation paragraphs. `config` = `sample.prompt`. `--prompt` forces config. |
| `prompt_words` | int | `8` | Words taken from a validation passage. |
| `output_dir` | string | `data/viz` | Unused when experiments are on; files land in the run `viz/`. |

---

## `logging`

| Key | Type | Default | How to use |
|---|---|---|---|
| `level` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | `INFO` | Loguru level. CLI `--log-level`. |
| `log_file` | string \| `null` | `logs/ha_llm.log` | Redirected to `.../logs/train.log` under the experiment. `null` = stderr only. |
| `tensorboard` | bool | `true` | Write scalars. |
| `tensorboard_dir` | string | `runs` | Redirected to the run `tb/`. `tensorboard --logdir data/experiments`. |

Scalars: `train/loss`, `train/lr`, `train/grad_norm`, `eval/*`.

---

## `experiment`

| Key | Type | Default | How to use |
|---|---|---|---|
| `enabled` | bool | `true` | If `true`, checkpoints/logs/tb/viz live under `root`. Tests set `false`. |
| `root` | string | `data/experiments` | Parent directory. |
| `name` | string \| `null` | `null` | Folder name. `null` → `<variant>_<YYYYMMDD_HHMMSS>`. CLI `--experiment NAME` reuses a folder. |
| `source_yaml` | string \| `null` | set at load | Path of the YAML you passed; do not set by hand. |

Layout:

```
data/experiments/<variant>/<variant>_<stamp>/
  config.yaml
  configs/
  checkpoints/last.pt
  logs/train.log
  tb/
  viz/
latest -> <that folder>
```

---

## Variant YAML cheat sheet

| File | `variant` | `arch` | `attn_type` | Extra |
|---|---|---|---|---|
| `configs/autoregressive/small.yaml` | `autoregressive` | `transformer` (from base) | `causal` | `use_time_cond: false` |
| `configs/diffusion/small.yaml` | `diffusion` | `lgt` | `bidirectional` | `use_time_cond: true` |
| `configs/block_diffusion/small.yaml` | `block_diffusion` | `lgt` | `block_causal` | `block_size: 16`, `use_time_cond: true` |
| `configs/flow_matching/small.yaml` | `flow_matching` | `lgt` | `bidirectional` | `use_time_cond: true` |
| `configs/mtp/small.yaml` | `mtp` | `transformer` | `causal` | `n_mtp_heads: 5` (stub) |

To try DiT on block diffusion, only change:

```yaml
# configs/block_diffusion/small.yaml
model:
  arch: dit
```

---

## CLI vs YAML

| Command | Required | Useful flags |
|---|---|---|
| `ha-llm-train` | `--config` | `--no-resume` new run; `--resume` / `--experiment NAME`; `--epochs` / `--steps`; `--viz` |
| `ha-llm-sample` | `--config` | `--experiment`, `--prompt`, `--checkpoint` |
| `ha-llm-eval` | `--config` | `--experiment`, `--metrics a,b` |
| `ha-llm-viz` | `--config` | `--experiment`, `--prompt`, `--output` |

Sample / eval / viz attach to `latest` unless `--experiment` is set. They do not create a new run folder.

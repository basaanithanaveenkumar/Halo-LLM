# Library layout

`hale_llm` is split so other projects (VLM, world models, custom LMs) can import pieces without the training CLI.

```
hale_llm/
  core/           components, transformer stacks, token backbones
  losses/         registered token losses (ce, focal, label_smoothing, kl)
  metrics/llm/    loss, perplexity, bits/token, accuracy, top-k
  dataloader/     tokenizer, windows, HF / overfit sources, collate registry
  evaluation/     Evaluator over a loader
  inference/      session, generate, encode_prompt
  models/         paradigm plugins (AR, diffusion, …)
```

## Losses

```python
from hale_llm.losses import token_nll, register_token_loss, TOKEN_LOSSES

nll = token_nll(logits, targets, loss_type="focal", focal_gamma=2.0, reduction="mean")
nll = token_nll(logits, targets, loss_type="label_smoothing", label_smoothing=0.1)

@register_token_loss("my_loss")
def my_loss(logits, targets, *, ignore_index=-100, **kwargs):
    ...
```

YAML: `train.loss_type: ce | focal | label_smoothing` plus `focal_gamma`, `focal_alpha`, `label_smoothing`.

Variant losses (`@register_loss("diffusion")`) still live under `models_*`; they call `model_token_nll`.

## Metrics

```python
from hale_llm.metrics.llm import LossMetric, PerplexityMetric, TokenAccuracyMetric, TopKAccuracyMetric
from hale_llm.core.registry import register_metric

@register_metric("my_metric")
class MyMetric:
    ...
```

YAML `eval.metrics`: `loss`, `perplexity`, `bits_per_token`, `token_accuracy`, `topk_accuracy`, plus variant `masked_accuracy`.

## Data

```python
from hale_llm.dataloader import DataModule, sliding_windows, setup_tokenizer, register_collate
```

Sources: `dataloader/sources/huggingface.py`, `dataloader/sources/overfit.py`.

## Eval / inference

```python
from hale_llm.evaluation import Evaluator, evaluate
from hale_llm.inference import load_session, generate, encode_prompt
```

Core stacks (no tokenizer): see [core.md](core.md).

## Registries

Two primitives in `hale_llm.core.registry`:

| Type | Use when | Examples |
|---|---|---|
| `NamedRegistry` | one config name maps to one implementation | model, loss, sampler, optimizer, collate, token loss, backbone, attention, ffn |
| `VariantRegistry` | one config name may have variant-specific implementations | metrics (`loss` is generic; `masked_accuracy` differs per diffusion variant) |

Training plugins register via decorators (`@register_variant`, `@register_loss`, …). Import `hale_llm.models` (or the relevant subpackage) so registration runs before lookup.

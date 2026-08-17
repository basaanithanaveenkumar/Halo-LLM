# Usage

```bash
uv sync
uv run ha-llm-train --config configs/block_diffusion/small.yaml --no-resume --viz
uv run ha-llm-sample --config configs/block_diffusion/small.yaml
uv run ha-llm-eval --config configs/block_diffusion/small.yaml
uv run ha-llm-viz --config configs/block_diffusion/small.yaml
tensorboard --logdir data/experiments
```

Configs inherit [`configs/base.yaml`](../configs/base.yaml). Full parameter list: [config.md](config.md).

# hale_llm.core

Thin compatibility namespace over the **`hale-blocks`** package (`hale_core`).

Prefer importing from `hale_core` directly:

```bash
uv add "hale-blocks @ git+https://github.com/basaanithanaveenkumar/HaleBlocks.git"
```

```python
from hale_core.nn.layers import build_attention
from hale_core.nn.backbones import build_backbone
from hale_core.registry import register_model
```

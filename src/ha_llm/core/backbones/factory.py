"""Open/closed backbone factory. New stacks register here; callers use build_backbone."""

from __future__ import annotations

from ha_llm.core.registry import NamedRegistry

BACKBONES = NamedRegistry("backbone")


def register_backbone(name: str):
    def deco(cls):
        BACKBONES.add(name, cls)
        return cls

    return deco


def backbone_kwargs(vocab_size: int, model_cfg, **overrides) -> dict:
    kwargs = {
        "vocab_size": vocab_size,
        "d_model": model_cfg.d_model,
        "n_heads": model_cfg.n_heads,
        "n_layers": model_cfg.n_layers,
        "d_ff": model_cfg.d_ff,
        "max_length": model_cfg.max_length,
        "dropout": model_cfg.dropout,
        "attn_type": model_cfg.attn_type,
        "use_time_cond": model_cfg.use_time_cond,
        "block_size": model_cfg.block_size,
        "attn_impl": model_cfg.attn_impl,
        "n_kv_heads": model_cfg.n_kv_heads,
        "ffn_type": model_cfg.ffn_type,
        "moe_num_experts": model_cfg.moe_num_experts,
        "moe_top_k": model_cfg.moe_top_k,
        "moe_num_shared": model_cfg.moe_num_shared,
        "arch": model_cfg.arch,
        "sliding_window": model_cfg.sliding_window,
        "local_global_ratio": model_cfg.local_global_ratio,
        "rope_theta_local": model_cfg.rope_theta_local,
        "rope_theta_global": model_cfg.rope_theta_global,
        "p_rope": model_cfg.p_rope,
        "qk_norm": model_cfg.qk_norm,
    }
    kwargs.update(overrides)
    return kwargs


def build_backbone(vocab_size: int, model_cfg=None, **overrides):
    if model_cfg is not None:
        kwargs = backbone_kwargs(vocab_size, model_cfg, **overrides)
    else:
        kwargs = {"vocab_size": vocab_size, **overrides}
        kwargs.setdefault("arch", "transformer")
    return BACKBONES.get(kwargs["arch"])(**kwargs)


def _register_builtins() -> None:
    from ha_llm.core.backbones.dit import DiTBackbone
    from ha_llm.core.backbones.gpt import GPTBackbone
    from ha_llm.core.backbones.lgt import LGTBackbone

    if "transformer" in BACKBONES:
        return
    register_backbone("default")(GPTBackbone)
    register_backbone("transformer")(GPTBackbone)
    register_backbone("lgt")(LGTBackbone)
    register_backbone("dit")(DiTBackbone)


_register_builtins()

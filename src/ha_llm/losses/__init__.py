"""Token-level losses as a library. Task losses (AR, diffusion) register separately."""

from ha_llm.losses.registry import TOKEN_LOSSES, get_token_loss, register_token_loss
from ha_llm.losses.token import loss_settings, model_token_nll, token_nll

__all__ = [
    "TOKEN_LOSSES",
    "register_token_loss",
    "get_token_loss",
    "token_nll",
    "model_token_nll",
    "loss_settings",
]

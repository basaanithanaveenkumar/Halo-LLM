"""Import token-loss implementations so `@register_token_loss` runs."""

from hale_llm.losses.functions import ce as _ce  # noqa: F401
from hale_llm.losses.functions import focal as _focal  # noqa: F401
from hale_llm.losses.functions import kl as _kl  # noqa: F401
from hale_llm.losses.functions import label_smoothing as _ls  # noqa: F401

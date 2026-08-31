"""Import every models_* subpackage so decorators fire.

Adding a paradigm: new folder `models/models_<name>/`, one import here, one YAML under configs/.
"""

from hale_llm.models import models_autoregressive as _ar  # noqa: F401
from hale_llm.models import models_block_diffusion as _bd  # noqa: F401
from hale_llm.models import models_diffusion as _diff  # noqa: F401
from hale_llm.models import models_flow_matching as _fm  # noqa: F401
from hale_llm.models import models_mtp as _mtp  # noqa: F401

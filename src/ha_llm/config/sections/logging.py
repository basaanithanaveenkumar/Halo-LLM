from typing import Literal

from ha_llm.config.sections.common import StrictModel


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str | None = "logs/ha_llm.log"
    tensorboard: bool = True
    tensorboard_dir: str = "runs"

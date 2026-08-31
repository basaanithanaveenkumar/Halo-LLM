"""Unified experiment logging supporting TensorBoard and Weights & Biases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


class ExperimentLogger:
    """
    Unified logger for experiment tracking with multiple backends.
    
    Supports:
    - TensorBoard: Local visualization
    - Weights & Biases (W&B): Cloud-based experiment tracking
    """
    
    def __init__(
        self,
        *,
        tensorboard_enabled: bool = False,
        tensorboard_dir: str | None = None,
        wandb_enabled: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_tags: list[str] | None = None,
        run_name: str | None = None,
    ):
        self.tensorboard_enabled = tensorboard_enabled
        self.wandb_enabled = wandb_enabled
        self.tb_writer = None
        self.wandb_run = None
        
        # Initialize TensorBoard
        if tensorboard_enabled:
            self._init_tensorboard(tensorboard_dir, run_name)
        
        # Initialize Weights & Biases
        if wandb_enabled:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or run_name,
                tags=wandb_tags,
            )
    
    def _init_tensorboard(self, log_dir: str | None, run_name: str | None) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("tensorboard requested but torch.utils.tensorboard not available")
            self.tensorboard_enabled = False
            return
        
        if not log_dir:
            logger.warning("tensorboard enabled but no log_dir; skipping")
            self.tensorboard_enabled = False
            return
        
        path = Path(log_dir)
        if run_name:
            path = path / run_name
        path.mkdir(parents=True, exist_ok=True)
        
        self.tb_writer = SummaryWriter(log_dir=str(path))
        logger.info("tensorboard writing to {}", path.resolve())
        logger.debug("open tensorboard with: tensorboard --logdir {}", Path(log_dir).resolve())
    
    def _init_wandb(
        self,
        project: str | None,
        entity: str | None,
        name: str | None,
        tags: list[str] | None,
    ) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb
        except ImportError:
            logger.warning(
                "wandb requested but not installed; install with: pip install wandb"
            )
            self.wandb_enabled = False
            return
        
        if not project:
            logger.warning("wandb enabled but no project specified; skipping")
            self.wandb_enabled = False
            return
        
        try:
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                tags=tags or [],
                reinit=True,
            )
            logger.info(
                "wandb initialized: project={} entity={} name={}",
                project,
                entity or "default",
                name or "auto",
            )
        except Exception as e:
            logger.error("failed to initialize wandb: {}", e)
            self.wandb_enabled = False
    
    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        """Log scalar metrics to all enabled backends."""
        if self.tensorboard_enabled and self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning("failed to log to wandb: {}", e)
    
    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log hyperparameters to all enabled backends."""
        # Clean hyperparameters for logging
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            elif v is None:
                clean_hparams[k] = "none"
            else:
                clean_hparams[k] = str(v)
        
        # TensorBoard
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.add_hparams(
                clean_hparams,
                metrics or {"hparam/placeholder": 0.0},
            )
        
        # Weights & Biases
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.config.update(clean_hparams, allow_val_change=True)
            except Exception as e:
                logger.warning("failed to log hparams to wandb: {}", e)
    
    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Log text to all enabled backends."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            except Exception as e:
                logger.warning("failed to log text to wandb: {}", e)
    
    def log_image(self, tag: str, image, step: int = 0) -> None:
        """Log image to all enabled backends."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.add_image(tag, image, step)
        
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.log({tag: wandb.Image(image)}, step=step)
            except Exception as e:
                logger.warning("failed to log image to wandb: {}", e)
    
    def log_histogram(self, tag: str, values, step: int = 0) -> None:
        """Log histogram to all enabled backends."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
        
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.log({tag: wandb.Histogram(values)}, step=step)
            except Exception as e:
                logger.warning("failed to log histogram to wandb: {}", e)
    
    def watch_model(self, model, log_freq: int = 100) -> None:
        """Watch model gradients and parameters (W&B only)."""
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.watch(model, log="all", log_freq=log_freq)
                logger.debug("wandb watching model gradients")
            except Exception as e:
                logger.warning("failed to watch model in wandb: {}", e)
    
    def flush(self) -> None:
        """Flush all loggers."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.flush()
        
        # W&B flushes automatically
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            logger.debug("tensorboard writer closed")
        
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.finish()
                logger.debug("wandb run finished")
            except Exception as e:
                logger.warning("failed to close wandb: {}", e)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def make_experiment_logger(
    *,
    tensorboard_enabled: bool = False,
    tensorboard_dir: str | None = None,
    wandb_enabled: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
    run_name: str | None = None,
) -> ExperimentLogger:
    """
    Factory function to create an experiment logger.
    
    Args:
        tensorboard_enabled: Enable TensorBoard logging
        tensorboard_dir: Directory for TensorBoard logs
        wandb_enabled: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team)
        wandb_run_name: W&B run name
        wandb_tags: W&B tags for the run
        run_name: Common run name for both backends
    
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        tensorboard_enabled=tensorboard_enabled,
        tensorboard_dir=tensorboard_dir,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        run_name=run_name,
    )

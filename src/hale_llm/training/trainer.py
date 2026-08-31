"""Variant-agnostic training loop. Resolves model/loss by name; no variant branching."""

from __future__ import annotations

import torch
from loguru import logger
from tqdm import tqdm

from hale_llm.config.experiment import apply_experiment_layout
from hale_llm.config.schema import RunConfig
from hale_llm.core.checkpoint import CheckpointStore
from hale_llm.core.optimizers import build_optimizer
from hale_llm.core.registry import get_loss, get_variant
from hale_llm.core.tensors import count_parameters, log_model_summary, move_batch_to_device
from hale_llm.dataloader.data_module import DataModule
from hale_llm.evaluation.evaluator import Evaluator
from hale_llm.training.distributed import (
    DistributedConfig,
    cleanup_distributed,
    is_main_process,
    reduce_dict,
    setup_distributed,
    wrap_model_parallel,
)
from hale_llm.training.schedule import TrainSchedule, resolve_schedule
from hale_llm.utils.device import get_device
from hale_llm.utils.experiment_logger import ExperimentLogger, make_experiment_logger
from hale_llm.utils.logging import setup_logging


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        tokenizer=None,
        *,
        data_module: DataModule | None = None,
        checkpoints: CheckpointStore | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.cfg = cfg
        self._tokenizer = tokenizer
        self.data = data_module
        self.checkpoints = checkpoints or CheckpointStore()
        
        # Initialize distributed training
        self.dist_config = setup_distributed(
            strategy=cfg.train.parallel_strategy,
            backend=cfg.train.distributed_backend,
        )

        # Set device (may be overridden by distributed setup)
        self.device = self._get_device(cfg.device)
        
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.model = None
        self.opt = None
        self.loss_fn = None
        self.logger: ExperimentLogger | None = None
        self.losses: list[float] = []
        self.global_step = 0
        self.start_epoch = 0
        self.schedule: TrainSchedule | None = None
    
    def _get_device(self, device_str: str | None) -> torch.device:
        """Get device, accounting for distributed training."""
        if self.dist_config.strategy in ("ddp", "fsdp") and self.dist_config.is_initialized:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.dist_config.local_rank}")
                torch.cuda.set_device(device)
                return device
        return torch.device(get_device(device_str))

    def fit(self) -> tuple:
        cfg = self.cfg
        apply_experiment_layout(cfg, create=True)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        torch.manual_seed(cfg.train.seed)

        if self.data is None:
            self.data = DataModule(cfg, self._tokenizer)
        self.tokenizer = self.data.tokenizer
        if self.evaluator is None:
            self.evaluator = Evaluator(cfg, device=self.device)

        vocab_size = len(self.tokenizer)
        logger.debug("vocab_size={}", vocab_size)

        try:
            model_cls = get_variant(cfg.variant)
            self.loss_fn = get_loss(cfg.variant)
        except KeyError:
            logger.error("unknown variant or missing loss/model: {}", cfg.variant)
            raise

        self.model = model_cls(vocab_size=vocab_size, cfg=cfg).to(self.device)
        n_params = count_parameters(self.model)
        
        # Wrap model with parallelization strategy
        self.model = wrap_model_parallel(self.model, self.dist_config, self.device)
        
        # Log model summary only on main process
        if is_main_process(self.dist_config):
            log_model_summary(self.model, title=f"model summary  variant={cfg.variant}")

        loader = self.data.train_loader()
        if len(loader) == 0:
            logger.error("empty dataloader; check batch_size / drop_last / dataset size")
            raise RuntimeError("empty dataloader")

        self.schedule = TrainSchedule.from_config(cfg, len(loader))
        n_epochs, total_steps = self.schedule.n_epochs, self.schedule.total_steps
        
        # Log training info only on main process
        if is_main_process(self.dist_config):
            logger.info(
                "starting train variant={} device={} parallel_strategy={} world_size={} "
                "epochs={} steps={} batches/epoch={} lr={} batch_size={} loss_type={}",
                cfg.variant,
                self.device,
                cfg.train.parallel_strategy,
                self.dist_config.world_size,
                n_epochs,
                total_steps,
                len(loader),
                cfg.train.lr,
                cfg.train.batch_size,
                cfg.train.loss_type,
            )
            logger.debug("full config: {}", cfg.model_dump())

        self.opt = build_optimizer(
            cfg.train.optimizer_type,
            self.model.parameters(),
            {
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "grad_clip": cfg.train.grad_clip,
            },
        )

        self._maybe_resume(n_epochs)

        # Initialize experiment logger (only main process logs)
        run_name = None if cfg.experiment.enabled else cfg.variant
        if cfg.experiment.enabled and cfg.experiment.name:
            run_name = cfg.experiment.name
        
        self.logger = make_experiment_logger(
            tensorboard_enabled=cfg.logging.tensorboard and is_main_process(self.dist_config),
            tensorboard_dir=cfg.logging.tensorboard_dir,
            wandb_enabled=cfg.logging.wandb and is_main_process(self.dist_config),
            wandb_project=cfg.logging.wandb_project,
            wandb_entity=cfg.logging.wandb_entity,
            wandb_run_name=cfg.logging.wandb_run_name,
            wandb_tags=cfg.logging.wandb_tags,
            run_name=run_name,
        )
        self.logger.log_hparams(
            {
                "variant": cfg.variant,
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
                "d_model": cfg.model.d_model,
                "n_layers": cfg.model.n_layers,
                "n_heads": cfg.model.n_heads,
                "max_length": cfg.model.max_length,
                "epochs": n_epochs,
                "steps": total_steps,
                "loss_type": cfg.train.loss_type,
                "parallel_strategy": cfg.train.parallel_strategy,
                "world_size": self.dist_config.world_size,
            },
        )
        self.logger.log_text("config", str(cfg.model_dump()))
        self.logger.log_scalars(0, {"model/n_params": float(n_params)})
        
        # Watch model with W&B if enabled
        if cfg.logging.wandb and cfg.logging.wandb_watch_model:
            self.logger.watch_model(self.model, log_freq=cfg.logging.wandb_log_freq)

        self.losses = []
        try:
            self._run_epochs(loader, vocab_size)
        except Exception:
            logger.exception("training crashed at last_step={}", self.global_step)
            if self.logger is not None:
                self.logger.close()
            cleanup_distributed(self.dist_config)
            raise

        # Only main process saves checkpoint
        if is_main_process(self.dist_config):
            self._save(epoch=n_epochs - 1, vocab_size=vocab_size)
            logger.info("checkpoint saved to {}", cfg.train.checkpoint_path)
        
        if self.logger is not None:
            if self.losses:
                self.logger.log_scalars(self.global_step, {"train/final_loss": self.losses[-1]})
            self.logger.close()
        
        if is_main_process(self.dist_config):
            logger.info(
                "training finished variant={} epochs={} steps={} final_loss={:.4f}",
                cfg.variant,
                n_epochs,
                self.global_step,
                self.losses[-1] if self.losses else float("nan"),
            )
        
        # Clean up distributed resources
        cleanup_distributed(self.dist_config)
        
        return self.model, self.tokenizer, self.losses

    def _maybe_resume(self, n_epochs: int) -> None:
        cfg = self.cfg
        ckpt_path = cfg.train.checkpoint_path
        self.start_epoch = 0
        self.global_step = 0
        if cfg.train.resume and self.checkpoints.exists(ckpt_path):
            if is_main_process(self.dist_config):
                logger.info("resume=true; loading {}", ckpt_path)
            ckpt = self.checkpoints.load(ckpt_path, self.device)
            
            # Handle DDP/FSDP wrapped models
            model_to_load = self.model
            if hasattr(self.model, "module"):
                model_to_load = self.model.module
            
            model_to_load.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.opt.load_state_dict(ckpt["optimizer_state_dict"])
                if is_main_process(self.dist_config):
                    logger.debug("restored optimizer state")
            else:
                if is_main_process(self.dist_config):
                    logger.warning("checkpoint has no optimizer_state_dict; continuing with fresh optimizer")
            self.start_epoch = int(ckpt.get("epoch", -1)) + 1
            self.global_step = int(ckpt.get("step", 0))
            if is_main_process(self.dist_config):
                logger.info("resumed from epoch={} step={}", self.start_epoch, self.global_step)
                if self.start_epoch >= n_epochs:
                    logger.warning(
                        "checkpoint epoch {} already past configured epochs={}; nothing to train. "
                        "use --no-resume to start over",
                        self.start_epoch,
                        n_epochs,
                    )
        elif cfg.train.resume:
            if is_main_process(self.dist_config):
                logger.info("resume=true but no checkpoint at {}; starting from scratch", ckpt_path)
        else:
            if is_main_process(self.dist_config):
                logger.info("resume=false; ignoring any checkpoint at {}", ckpt_path)

    def _save(self, *, epoch: int, vocab_size: int) -> None:
        # Extract model state from DDP/FSDP wrapper if needed
        model_to_save = self.model
        if hasattr(self.model, "module"):
            model_to_save = self.model.module
        
        self.checkpoints.save(
            self.cfg.train.checkpoint_path,
            model_state_dict=model_to_save.state_dict(),
            optimizer_state_dict=self.opt.state_dict(),
            variant=self.cfg.variant,
            vocab_size=vocab_size,
            epoch=epoch,
            step=self.global_step,
            config=self.cfg.model_dump(),
        )

    def _run_epochs(self, loader, vocab_size: int) -> None:
        cfg = self.cfg
        n_epochs = self.schedule.n_epochs
        total_steps = self.schedule.total_steps
        for epoch in range(self.start_epoch, n_epochs):
            self.model.train()
            epoch_losses: list[float] = []
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=True)
            for batch in pbar:
                if self.global_step >= total_steps:
                    break
                batch = move_batch_to_device(batch, self.device)
                if self.global_step == 0:
                    logger.debug(
                        "batch keys={} shapes={}",
                        list(batch),
                        {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
                    )

                loss = self.loss_fn(self.model, batch)
                if not torch.isfinite(loss):
                    logger.error("non-finite loss at step {}: {}", self.global_step, loss.item())
                    raise FloatingPointError(f"non-finite loss at step {self.global_step}")

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                self.opt.step()
                lr = self.opt.param_groups[0]["lr"]
                self.losses.append(loss.item())
                epoch_losses.append(loss.item())

                self.logger.log_scalars(
                    self.global_step,
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/epoch": float(epoch),
                    },
                )
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if self.global_step % cfg.train.log_every == 0:
                    logger.info(
                        "epoch {}/{} step {}/{} loss={:.4f} lr={:.2e} grad_norm={:.3f}",
                        epoch + 1,
                        n_epochs,
                        self.global_step,
                        total_steps,
                        loss.item(),
                        lr,
                        float(grad_norm),
                    )
                self.global_step += 1
                self._maybe_visualize(tag=f"step{self.global_step}")

            epoch_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
            self.logger.log_scalars(epoch, {"train/epoch_loss": epoch_mean})
            
            if is_main_process(self.dist_config):
                logger.info(
                    "epoch {}/{} done mean_loss={:.4f} steps_this_epoch={}",
                    epoch + 1,
                    n_epochs,
                    epoch_mean,
                    len(epoch_losses),
                )
            
            if cfg.eval.every_n_epochs and (epoch + 1) % cfg.eval.every_n_epochs == 0:
                eval_loader = self.data.val_loader(loader)
                eval_metrics = self.evaluator.run(self.model, eval_loader)
                
                # Reduce metrics across all processes
                eval_metrics = reduce_dict(eval_metrics, self.dist_config, average=True)
                
                self.logger.log_scalars(
                    epoch,
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                )
            
            # Only main process does visualization
            if is_main_process(self.dist_config):
                self._maybe_visualize(tag=f"epoch{epoch + 1}", by_epoch=True)
            
            # Only main process saves checkpoints
            if cfg.train.checkpoint_every_epoch and is_main_process(self.dist_config):
                self._save(epoch=epoch, vocab_size=vocab_size)

    def _maybe_visualize(self, *, tag: str, by_epoch: bool = False) -> None:
        cfg = self.cfg
        if not cfg.viz.enabled:
            return
        if by_epoch:
            n = cfg.viz.every_n_epochs
            if not n:
                return
            epoch_n = int(tag.replace("epoch", ""))
            if epoch_n % n != 0:
                return
        else:
            n = cfg.viz.every_n_steps
            if not n or self.global_step % n != 0:
                return
        from hale_llm.visualization.run import run_visualization

        out = f"{cfg.viz.output_dir.rstrip('/')}/{cfg.variant}_{tag}.gif"
        try:
            was_training = self.model.training
            run_visualization(
                self.model,
                self.tokenizer,
                cfg,
                self.device,
                out,
                train=True,
                prompt=self.data.viz_prompt(self.global_step),
            )
            if was_training:
                self.model.train()
            logger.info("saved training viz {}", out)
        except Exception:
            logger.exception("visualization failed at {}; training continues", tag)


def train(cfg: RunConfig, tokenizer=None) -> tuple:
    return Trainer(cfg, tokenizer).fit()

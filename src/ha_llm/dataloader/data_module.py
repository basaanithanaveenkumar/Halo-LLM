"""Build train/val loaders from config. Trainer depends on this, not on Hub details."""

from __future__ import annotations

from torch.utils.data import DataLoader

from ha_llm.config.schema import RunConfig
from ha_llm.dataloader.loaders import make_dataloader
from ha_llm.dataloader.prompts import collect_dataset_passages, dataset_prompt
from ha_llm.dataloader.sources.overfit import make_overfit_loader, uses_overfit
from ha_llm.dataloader.tokenizer import setup_tokenizer
from loguru import logger


class DataModule:
    def __init__(self, cfg: RunConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer or setup_tokenizer(cfg.data.tokenizer_name)
        self._viz_passages: list[str] | None = None

    def viz_prompt(self, seed: int = 0) -> str:
        if self.cfg.viz.prompt_source != "dataset":
            return self.cfg.sample.prompt
        if self._viz_passages is None:
            self._viz_passages = collect_dataset_passages(self.cfg, min_words=self.cfg.viz.prompt_words)
        return dataset_prompt(self.cfg, seed=seed, passages=self._viz_passages)

    def train_loader(self) -> DataLoader:
        cfg = self.cfg
        if uses_overfit(cfg):
            if not cfg.data.overfit_text:
                raise ValueError("data.source=overfit requires data.overfit_text")
            logger.warning("overfit mode enabled; HuggingFace loader is bypassed")
            logger.debug("overfit_text={!r} copies={}", cfg.data.overfit_text, cfg.data.n_overfit_copies)
            return make_overfit_loader(
                cfg.variant,
                self.tokenizer,
                cfg.data.overfit_text,
                cfg.model.max_length,
                cfg.train.batch_size,
                cfg.data.n_overfit_copies,
            )
        return make_dataloader(cfg, split="train")

    def val_loader(self, train_loader: DataLoader | None = None) -> DataLoader:
        if uses_overfit(self.cfg):
            if train_loader is None:
                return self.train_loader()
            return train_loader
        return make_dataloader(self.cfg, split="validation")

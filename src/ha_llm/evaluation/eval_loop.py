"""Compatibility shim. Prefer `ha_llm.evaluation.evaluator`."""

from ha_llm.evaluation.evaluator import Evaluator, evaluate

__all__ = ["Evaluator", "evaluate"]

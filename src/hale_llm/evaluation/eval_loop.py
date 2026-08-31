"""Compatibility shim. Prefer `hale_llm.evaluation.evaluator`."""

from hale_llm.evaluation.evaluator import Evaluator, evaluate

__all__ = ["Evaluator", "evaluate"]

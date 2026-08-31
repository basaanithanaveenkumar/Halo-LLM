"""Discrete flow interpolant helpers.

TODO: extract path construction (masking probability, weights) from model.py
when you add non-linear interpolants.
"""


def mask_prob(t):
    """Linear masking path: P(masked | t) = 1 - t  (t=0 source, t=1 data)."""
    return 1.0 - t

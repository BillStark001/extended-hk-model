"""Terminal-state classification shared by microscopic analyses."""

from .classifier import (
    ClassifierOptions,
    classify_atomic_measure,
    classify_opinions,
)

__all__ = [
    "ClassifierOptions",
    "classify_atomic_measure",
    "classify_opinions",
]

"""
CLIFS: Cognitive Linguistic Identity Fusion Score

A cognition-informed Python framework for quantifying identity fusion from text.
"""

__version__ = "0.1.0"
__author__ = "Devin R. Wright"
__license__ = "Apache-2.0"

from clifs import clifs
from clifs.clifs import (
    load_model,
    load_models,
    load_regression_model,
)

__all__ = [
    "clifs",
    "load_model",
    "load_models",
    "load_regression_model",
    "__version__",
]
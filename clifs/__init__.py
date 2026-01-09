"""CLIFS: Cognitive Linguistic Identity Fusion Score.

A cognition-informed Python framework for quantifying identity fusion from text.
"""

__version__ = "0.1.0"
__author__ = "Devin R. Wright"
__license__ = "Apache-2.0"

from clifs.config import CLIFSConfig, PathSettings, RuntimeSettings
from clifs.models import ModelLoader, ModelRegistry
from clifs.predictors import (
    ClassificationPredictor,
    RegressionPredictor,
    EnsemblePredictor
)
from clifs.features import FeatureExtractor


# Convenience functions for simple usage
def predict_classification(df, known_groups=None, force_cpu=False):
    """Quick classification prediction.

    Args:
        df: DataFrame with 'text' column
        known_groups: Optional list of known target groups
        force_cpu: Force CPU usage

    Returns:
        DataFrame with predictions
    """
    config = CLIFSConfig.from_env()
    config.runtime.force_cpu = force_cpu
    config.runtime.setup_environment()

    loader = ModelLoader(config)
    models = loader.load_base_models()

    predictor = ClassificationPredictor(models, tuple(known_groups or []))
    return predictor.predict_batch(df['text'])


def predict_regression(df, known_groups=None, force_cpu=False):
    """Quick regression prediction.

    Args:
        df: DataFrame with 'text' column
        known_groups: Optional list of known target groups
        force_cpu: Force CPU usage

    Returns:
        DataFrame with predictions
    """
    config = CLIFSConfig.from_env()
    config.runtime.force_cpu = force_cpu
    config.runtime.setup_environment()

    loader = ModelLoader(config)
    models = loader.load_base_models()
    models = loader.add_regression(models)

    predictor = RegressionPredictor(models, tuple(known_groups or []))
    return predictor.predict_batch(df['text'])


__all__ = [
    "CLIFSConfig",
    "PathSettings",
    "RuntimeSettings",
    "ModelLoader",
    "ModelRegistry",
    "ClassificationPredictor",
    "RegressionPredictor",
    "EnsemblePredictor",
    "FeatureExtractor",
    "predict_classification",
    "predict_regression",
    "__version__",
]
"""Smoke tests - quick sanity checks that don't require model loading.

Run these first to verify basic functionality before running full integration tests.
"""

import pytest
from clifs import CLIFSConfig


def test_imports():
    """Test that all main modules can be imported."""
    from clifs import (
        CLIFSConfig,
        PathSettings,
        RuntimeSettings,
        ModelLoader,
        ModelRegistry,
        ClassificationPredictor,
        RegressionPredictor,
        EnsemblePredictor,
        FeatureExtractor,
        predict_classification,
        predict_regression,
    )
    assert True


def test_config_creation():
    """Test that configuration can be created."""
    config = CLIFSConfig.from_env()
    assert config is not None
    assert config.paths is not None
    assert config.runtime is not None


def test_paths_exist():
    """Test that required paths exist."""
    config = CLIFSConfig.from_env()

    assert config.paths.project_root.exists(), "Project root not found"
    assert config.paths.models_dir.exists(), "Models directory not found"
    assert config.paths.data_dir.exists(), "Data directory not found"


def test_model_files_exist():
    """Test that required model files exist."""
    config = CLIFSConfig.from_env()

    # Check ModernBERT model
    assert config.paths.mbert_model_path.exists(), \
        f"ModernBERT model not found at {config.paths.mbert_model_path}"

    # Check RF models
    assert config.paths.rf_models_dir.exists(), \
        f"RF models directory not found at {config.paths.rf_models_dir}"

    assert (config.paths.rf_models_dir / "best_model_rf_aug.joblib").exists(), \
        "Classification model not found"

    assert (config.paths.rf_models_dir / "augmented_regression_best.joblib").exists(), \
        "Regression model not found"


def test_device_detection():
    """Test device detection."""
    config = CLIFSConfig.from_env()
    device = config.runtime.device

    assert device is not None
    assert str(device) in ["cpu", "cuda:0", "mps:0"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
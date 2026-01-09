"""Integration tests for CLIFS.

These tests verify that the entire pipeline works end-to-end.
They require models to be downloaded and may take several minutes to run.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from clifs import CLIFSConfig, ModelLoader, predict_classification, predict_regression
from clifs.predictors import ClassificationPredictor, RegressionPredictor


@pytest.fixture(scope="module")
def test_data():
    """Create test dataframe with sample texts."""
    data = {
        'text': [
            "I am proud to be an American. I love my country deeply.",
            "I attended that university but didn't enjoy it much.",
            "Our team is like family. We are one.",
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def test_csv(tmp_path_factory):
    """Create a temporary CSV file with test data."""
    tmpdir = tmp_path_factory.mktemp("test_data")
    csv_path = tmpdir / "test_input.csv"

    data = {
        'text': [
            "I am proud to be an American. I love my country deeply.",
            "I attended that university but didn't enjoy it much.",
            "Our team is like family. We are one.",
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="module")
def config():
    """Create configuration for tests."""
    config = CLIFSConfig.from_env()
    config.runtime.force_cpu = True  # Use CPU for tests
    config.runtime.setup_environment()
    return config


@pytest.fixture(scope="module")
def models(config):
    """Load models once for all tests."""
    loader = ModelLoader(config)
    return loader.load_base_models()


class TestConfiguration:
    """Test configuration management."""

    def test_config_from_env(self):
        """Test loading configuration from environment."""
        config = CLIFSConfig.from_env()
        assert config.paths.project_root.exists()
        assert config.paths.models_dir.exists()
        assert config.runtime.device is not None

    def test_path_settings(self):
        """Test path configuration."""
        config = CLIFSConfig.from_env()
        assert config.paths.mbert_model_path.exists(), "ModernBERT model not found"
        assert config.paths.rf_models_dir.exists(), "RF models directory not found"

    def test_runtime_settings(self):
        """Test runtime settings."""
        config = CLIFSConfig.from_env()
        config.runtime.force_cpu = True
        assert str(config.runtime.device) == "cpu"


class TestModelLoading:
    """Test model loading."""

    def test_base_models_load(self, config):
        """Test that base models load successfully."""
        loader = ModelLoader(config)
        models = loader.load_base_models()

        assert models.mbert_fine_tuned is not None
        assert models.mbert_base is not None
        assert models.tokenizer is not None
        assert models.nlp is not None
        assert models.sbert_model is not None
        assert models.clifs_rf is not None
        assert models.device is not None

    def test_regression_model_load(self, config, models):
        """Test loading regression model."""
        loader = ModelLoader(config)
        models_with_reg = loader.add_regression(models)

        assert models_with_reg.clifs_rf_regression is not None


class TestFeatureExtraction:
    """Test feature extraction."""

    def test_extract_features(self, models):
        """Test extracting features from text."""
        from clifs.features import FeatureExtractor

        extractor = FeatureExtractor(models, known_groups=('country',))
        text = "I love my country."

        features = extractor.extract_all(text)

        # Should have 768 (sbert) + 3 (mbert) + 4 (mlmif) + 3 (uai) + 2 (vri) = 780 dims
        assert features.shape == (780,)
        assert not pd.isna(features).any()

    def test_individual_extractors(self, models):
        """Test individual feature extractors."""
        from clifs.features import FeatureExtractor

        extractor = FeatureExtractor(models, known_groups=('country',))
        text = "I love my country."

        # Test each extractor
        sbert = extractor.extract_sbert(text)
        assert sbert.shape == (768,)

        mbert = extractor.extract_mbert_probs(text)
        assert mbert.shape == (3,)

        mlmif = extractor.extract_mlmif(text)
        assert mlmif.shape == (4,)

        uai = extractor.extract_uai(text)
        assert uai.shape == (3,)

        vri = extractor.extract_vri(text)
        assert vri.shape == (2,)


class TestClassification:
    """Test classification prediction."""

    def test_classification_single(self, models):
        """Test classification on single text."""
        predictor = ClassificationPredictor(models, known_groups=('country', 'america'))

        text = "I am proud to be an American. I love my country deeply."
        result = predictor.predict_single(text)

        assert 'clifs' in result
        assert 'clifs_fusion_numeric' in result
        assert result['clifs'] in ['low', 'medium', 'high']
        assert isinstance(result['clifs_fusion_numeric'], (int, float))

    def test_classification_batch(self, models, test_data):
        """Test classification on batch of texts."""
        predictor = ClassificationPredictor(models, known_groups=('country', 'america'))

        results = predictor.predict_batch(test_data['text'])

        assert len(results) == len(test_data)
        assert 'clifs' in results.columns
        assert 'clifs_fusion_numeric' in results.columns

    def test_classification_convenience(self, test_data):
        """Test convenience function for classification."""
        results = predict_classification(
            test_data,
            known_groups=['country', 'america'],
            force_cpu=True
        )

        assert len(results) == len(test_data)
        assert 'clifs' in results.columns


class TestRegression:
    """Test regression prediction."""

    def test_regression_single(self, config, models):
        """Test regression on single text."""
        loader = ModelLoader(config)
        models_with_reg = loader.add_regression(models)

        predictor = RegressionPredictor(models_with_reg, known_groups=('country', 'america'))

        text = "I am proud to be an American. I love my country deeply."
        result = predictor.predict_single(text)

        assert 'clifs_fusion_r' in result
        assert isinstance(result['clifs_fusion_r'], float)
        assert 1.0 <= result['clifs_fusion_r'] <= 7.0  # VIFS scale

    def test_regression_batch(self, config, models, test_data):
        """Test regression on batch of texts."""
        loader = ModelLoader(config)
        models_with_reg = loader.add_regression(models)

        predictor = RegressionPredictor(models_with_reg, known_groups=('country', 'america'))

        results = predictor.predict_batch(test_data['text'])

        assert len(results) == len(test_data)
        assert 'clifs_fusion_r' in results.columns

    def test_regression_convenience(self, test_data):
        """Test convenience function for regression."""
        results = predict_regression(
            test_data,
            known_groups=['country', 'america'],
            force_cpu=True
        )

        assert len(results) == len(test_data)
        assert 'clifs_fusion_r' in results.columns


class TestCLI:
    """Test CLI functionality."""

    def test_cli_classification(self, test_csv, tmp_path):
        """Test CLI classification mode."""
        import subprocess

        output_path = tmp_path / "output.csv"

        result = subprocess.run(
            [
                "python", "-m", "clifs",
                "--input", str(test_csv),
                "--output", str(output_path),
                "--groups", "country", "america",
                "--cpu"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_path.exists()

        # Verify output
        df = pd.read_csv(output_path)
        assert 'clifs' in df.columns
        assert len(df) == 3

    def test_cli_regression(self, test_csv, tmp_path):
        """Test CLI regression mode."""
        import subprocess

        output_path = tmp_path / "output_regression.csv"

        result = subprocess.run(
            [
                "python", "-m", "clifs",
                "--input", str(test_csv),
                "--output", str(output_path),
                "--groups", "country", "america",
                "--regression",
                "--cpu"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_path.exists()

        # Verify output
        df = pd.read_csv(output_path)
        assert 'clifs_fusion_r' in df.columns
        assert len(df) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self, models):
        """Test handling of empty text."""
        predictor = ClassificationPredictor(models)

        try:
            result = predictor.predict_single("")
            # Should either handle gracefully or raise clear error
            assert isinstance(result, dict)
        except Exception as e:
            # Should raise a clear, specific error
            assert str(e)  # Error message exists

    def test_very_long_text(self, models):
        """Test handling of very long text."""
        predictor = ClassificationPredictor(models)

        long_text = "I love my country. " * 1000  # Very long text
        result = predictor.predict_single(long_text)

        assert 'clifs' in result
        assert result['clifs'] in ['low', 'medium', 'high']

    def test_special_characters(self, models):
        """Test handling of special characters."""
        predictor = ClassificationPredictor(models)

        text = "I ❤️ my country! #proud @america 🇺🇸"
        result = predictor.predict_single(text)

        assert 'clifs' in result

    def test_no_known_groups(self, models):
        """Test prediction without known groups."""
        predictor = ClassificationPredictor(models, known_groups=())

        text = "I love my country."
        result = predictor.predict_single(text)

        assert 'clifs' in result


class TestPredictionConsistency:
    """Test that predictions are consistent."""

    def test_same_text_same_result(self, models):
        """Test that same text produces same result."""
        predictor = ClassificationPredictor(models, known_groups=('country',))

        text = "I am proud to be an American."

        result1 = predictor.predict_single(text)
        result2 = predictor.predict_single(text)

        assert result1['clifs'] == result2['clifs']
        assert result1['clifs_fusion_numeric'] == result2['clifs_fusion_numeric']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
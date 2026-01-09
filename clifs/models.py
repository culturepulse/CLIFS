"""Model management without globals."""

from dataclasses import dataclass
from typing import Any, Optional, Dict
import torch
import joblib
import pandas as pd
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from clifs.config import CLIFSConfig
from clifs.core import masked_lm_identity_fusion as mlmif


@dataclass
class ModelRegistry:
    """Container for all loaded models - replaces global variables."""

    # Core models
    mbert_fine_tuned: Any
    mbert_base: Any
    tokenizer: Any
    nlp: Any
    sbert_model: SentenceTransformer
    clifs_rf: Any

    # Device
    device: torch.device

    # Optional models
    clifs_rf_regression: Optional[Any] = None
    sbert_rf: Optional[Any] = None

    # RAG data
    docs_df: Optional[pd.DataFrame] = None
    example_texts: Optional[Dict[str, str]] = None


class ModelLoader:
    """Loads models based on configuration."""

    def __init__(self, config: CLIFSConfig):
        self.config = config
        self.device = config.runtime.device

    def load_base_models(self) -> ModelRegistry:
        """Load models required for basic classification."""

        # Load SBERT
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
        sbert_model.to(self.device)

        # Load fine-tuned ModernBERT
        mbert_fine_tuned = AutoModelForSequenceClassification.from_pretrained(
            str(self.config.paths.mbert_model_path)
        )
        mbert_fine_tuned.to(self.device)
        mbert_fine_tuned.eval()

        # Load base ModernBERT
        mbert_base, tokenizer, nlp = mlmif.load_model_nlp_and_tokenizer(
            device=self.device
        )

        # Load Random Forest
        clifs_rf = joblib.load(
            self.config.paths.rf_models_dir / 'best_model_rf_aug.joblib'
        )

        return ModelRegistry(
            mbert_fine_tuned=mbert_fine_tuned,
            mbert_base=mbert_base,
            tokenizer=tokenizer,
            nlp=nlp,
            sbert_model=sbert_model,
            clifs_rf=clifs_rf,
            device=self.device
        )

    def add_regression(self, registry: ModelRegistry) -> ModelRegistry:
        """Add regression model to registry."""
        registry.clifs_rf_regression = joblib.load(
            self.config.paths.rf_models_dir / 'augmented_regression_best.joblib'
        )
        return registry

    def add_ensemble(self, registry: ModelRegistry) -> ModelRegistry:
        """Add ensemble models and data to registry."""

        # Load SBERT RF
        registry.sbert_rf = joblib.load(
            self.config.paths.rf_models_dir / 'sbert_classification_best_model_augmented.joblib'
        )

        # Load training data
        aug_train = self.config.paths.data_dir / 'ap_study1_augmented_finegrain_train.csv'
        docs_df = pd.read_csv(aug_train)
        registry.docs_df = docs_df

        # Extract examples
        registry.example_texts = {
            'low': docs_df[docs_df['fusion'] == docs_df['fusion'].min()].iloc[0]['write'],
            'medium': docs_df[docs_df['fusion'] == docs_df['fusion'].median()].iloc[0]['write'],
            'high': docs_df[docs_df['fusion'] == docs_df['fusion'].max()].iloc[0]['write']
        }

        return registry
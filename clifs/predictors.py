"""Prediction classes."""

import pandas as pd
from typing import Tuple, Dict, Any, Optional
from collections import Counter

from clifs.models import ModelRegistry
from clifs.features import FeatureExtractor


LABEL_MAP = {0: "high", 1: "low", 2: "medium"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


class BasePredictor:
    """Base predictor class."""

    def __init__(self, models: ModelRegistry, known_groups: Tuple[str, ...] = ()):
        self.models = models
        self.extractor = FeatureExtractor(models, known_groups)


class ClassificationPredictor(BasePredictor):
    """Classification predictor."""

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict fusion class for single text."""
        features = self.extractor.extract_all(text)
        prediction = self.models.clifs_rf.predict(features.reshape(1, -1))[0]

        return {
            'clifs': prediction,
            'clifs_fusion_numeric': LABEL_MAP_INV[prediction]
        }

    def predict_batch(self, texts: pd.Series) -> pd.DataFrame:
        """Predict for multiple texts."""
        results = [self.predict_single(text) for text in texts]
        return pd.DataFrame(results)


class RegressionPredictor(BasePredictor):
    """Regression predictor."""

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict continuous fusion score."""
        features = self.extractor.extract_all(text)
        score = self.models.clifs_rf_regression.predict(features.reshape(1, -1))[0]

        return {'clifs_fusion_r': score}

    def predict_batch(self, texts: pd.Series) -> pd.DataFrame:
        """Predict for multiple texts."""
        results = [self.predict_single(text) for text in texts]
        return pd.DataFrame(results)


class EnsemblePredictor(BasePredictor):
    """Ensemble predictor combining multiple models."""

    def __init__(
        self,
        models: ModelRegistry,
        known_groups: Tuple[str, ...],
        openai_client,
        deepseek_client
    ):
        super().__init__(models, known_groups)
        self.openai_client = openai_client
        self.deepseek_client = deepseek_client
        self._faiss_index = None  # Cache FAISS index

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Ensemble prediction."""
        from clifs.core import rag_fusion_classification as rag

        # CLIFS RF prediction
        features = self.extractor.extract_all(text)
        clifs_pred = self.models.clifs_rf.predict(features.reshape(1, -1))[0]

        # SBERT RF prediction
        sbert_features = self.extractor.extract_sbert(text).reshape(1, -1)
        sbert_pred = self.models.sbert_rf.predict(sbert_features)[0]

        # Build FAISS index (cached after first call)
        if self._faiss_index is None:
            self._faiss_index = rag.build_faiss_index(
                self.models.docs_df,
                self.models.sbert_model
            )

        # RAG predictions
        rag_4o_pred = rag.classify_text_rag(
            text=text,
            low_text=self.models.example_texts['low'],
            medium_text=self.models.example_texts['medium'],
            high_text=self.models.example_texts['high'],
            client=self.openai_client,
            index=self._faiss_index,
            model=self.models.sbert_model,
            rag_model='gpt-4o',
            docs_df=self.models.docs_df
        )

        rag_r1_pred = rag.classify_text_rag(
            text=text,
            low_text=self.models.example_texts['low'],
            medium_text=self.models.example_texts['medium'],
            high_text=self.models.example_texts['high'],
            client=self.deepseek_client,
            index=self._faiss_index,
            model=self.models.sbert_model,
            rag_model='deepseek-reasoner',
            docs_df=self.models.docs_df
        )

        # Hard voting
        vote = self._hard_vote(clifs_pred, sbert_pred, rag_4o_pred, rag_r1_pred)

        return {
            'sbert_rf': sbert_pred,
            'clifs': clifs_pred,
            'rag_4o': rag_4o_pred,
            'rag_r1': rag_r1_pred,
            'clifs_fusion': vote,
            'clifs_fusion_numeric': LABEL_MAP_INV[vote]
        }

    @staticmethod
    def _hard_vote(cl_rf, sb_rf, rag_4o, rag_r1) -> str:
        """Hard voting with tie-breaking."""
        votes = [cl_rf, sb_rf, rag_4o, rag_r1]
        counter = Counter(votes)
        most_common = counter.most_common()

        top_count = most_common[0][1]
        top_candidates = [label for label, count in most_common if count == top_count]

        # Tie-breaker: high > low > medium
        priority = {'high': 3, 'low': 2, 'medium': 1}
        top_candidates.sort(key=lambda x: priority[x], reverse=True)

        return top_candidates[0]
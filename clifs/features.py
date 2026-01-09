"""Feature extraction - pass models explicitly."""

import numpy as np
import torch
from typing import Tuple

from clifs.models import ModelRegistry
from clifs.core import masked_lm_identity_fusion as mlmif
from clifs.core import unquestioning_affiliation_index
from clifs.core import violence_risk_index_lite_clifs as vri
from clifs.core.uai_codes import affiliation, cogproc


class FeatureExtractor:
    """Extracts all features for CLIFS prediction."""

    def __init__(self, models: ModelRegistry, known_groups: Tuple[str, ...] = ()):
        self.models = models
        self.known_groups = known_groups

    def extract_mlmif(self, text: str) -> np.ndarray:
        """Extract MLMIF features."""
        ItoT = mlmif.compute_identity_to_target_score(
            text,
            self.known_groups,
            self.models.mbert_base,
            self.models.tokenizer,
            self.models.nlp,
            self.models.device
        )
        TtoI = mlmif.compute_target_to_identity_score(
            text,
            self.known_groups,
            self.models.mbert_base,
            self.models.tokenizer,
            self.models.nlp,
            self.models.device
        )
        KtoT = mlmif.compute_kin_fusion_score(
            text,
            self.known_groups,
            self.models.mbert_base,
            self.models.tokenizer,
            self.models.nlp,
            self.models.device
        )
        mlmif_score = mlmif.compute_fusion_proximity_score(ItoT, TtoI)

        return np.array([ItoT, TtoI, KtoT, mlmif_score])

    def extract_mbert_probs(self, text: str) -> np.ndarray:
        """Extract ModernBERT class probabilities."""
        inputs = self.models.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.models.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.models.mbert_fine_tuned(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        return probabilities.cpu().detach().numpy().flatten()

    def extract_sbert(self, text: str) -> np.ndarray:
        """Extract SBERT embeddings."""
        return self.models.sbert_model.encode(text)

    def extract_uai(self, text: str) -> np.ndarray:
        """Extract UAI features."""
        uai_aff = unquestioning_affiliation_index.calculate_word_counts(text, affiliation)
        uai_cp = unquestioning_affiliation_index.calculate_word_counts(text, cogproc)
        nUAI = unquestioning_affiliation_index.calculate_nuai(uai_aff, uai_cp)
        return np.array([uai_aff, uai_cp, nUAI])

    def extract_vri(self, text: str) -> np.ndarray:
        """Extract VRI features."""
        vri_dict = vri.process_text(text)
        return np.array([vri_dict['fusion'], vri_dict['identification']])

    def extract_all(self, text: str) -> np.ndarray:
        """Extract all features and concatenate."""
        return np.hstack([
            self.extract_sbert(text),           # 768 dims
            self.extract_mbert_probs(text),     # 3 dims
            self.extract_mlmif(text),           # 4 dims
            self.extract_uai(text),             # 3 dims
            self.extract_vri(text)              # 2 dims
        ])
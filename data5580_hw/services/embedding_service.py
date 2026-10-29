import numpy as np
import pandas as pd

from data5580_hw.models.prediction import Embedding, Prediction


class EmbeddingService:

    @staticmethod
    def create_embedding(prediction: Prediction) -> Prediction:
        if prediction.model is None or prediction.model._umap is None:
            model_name = prediction.model.name if prediction.model else "unknown"
            model_version = prediction.model.version if prediction.model else "unknown"
            raise ValueError(
                f"UMAP model is not configured for model {model_name} version {model_version}"
            )

        features = prediction.features
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = pd.DataFrame([features])

        try:
            raw = prediction.model._umap.predict(features_df)
        except Exception as e:
            raise ValueError(f"UMAP computation failed: {e}") from e

        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("UMAP computation failed: embedding was empty or non-numeric")

        prediction.embeddings = [
            Embedding(
                kind="inputs",
                values=[float(value) for value in values],
            )
        ]
        return prediction


embedding_service: EmbeddingService = EmbeddingService()

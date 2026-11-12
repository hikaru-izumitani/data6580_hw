from data5580_hw.models.prediction import Embedding, Prediction


class EmbeddingService:

    @staticmethod
    def create_embedding(prediction: Prediction) -> Prediction:
        if prediction.model is None or prediction.model._embedding is None:
            raise ValueError("UMAP computation failed: embedding model is not configured.")
        try:
            raw = prediction.model._embedding.predict(prediction.features)
        except Exception as error:
            raise ValueError(f"UMAP computation failed: {error}") from error
        
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if values and isinstance(values[0], (list, tuple)):
            values = list(values[0])
        values = [float(value) for value in values]
        if len(values) !=3:
            raise ValueError(f"UMAP computation failed: expected 3 values, got {len(values)}")



        prediction.embedding_values = Embedding(embeddings=values)
        return prediction


embedding_service: EmbeddingService = EmbeddingService()

from celery import shared_task

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

    @staticmethod
    @shared_task(bind=True)
    def generate_embedding(prediction_id: str) -> str:
        from data5580_hw.services.database.database_client import db

        prediction = Prediction.from_prediction_sql(prediction_id)

        # Perform embedding
        prediction = embedding_service.create_embedding(prediction)

        # Persist embedding
        embedding_sql = prediction.embedding_values.to_sql(prediction_id)

        db.session.add(embedding_sql)
        db.session.commit()

        return prediction_id


embedding_service: EmbeddingService = EmbeddingService()

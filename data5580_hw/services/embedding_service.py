# import json
# import logging
# import pandas as pd
# from celery import shared_task
# from data5580_hw.gateways import mlflow_gateway

from data5580_hw.models.prediction import Embedding, Prediction

# logger = logging.getLogger(__name__)


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

    # @staticmethod
    # @shared_task(
    #     autoretry_for=(Exception,),
    #     retry_backoff=True,
    #     retry_kwargs={"max_retries": 3},
    #     soft_time_limit=60,
    #     time_limit=90,
    # )
    # def generate_embedding(prediction_id: str) -> str:
    #     Moved to data5580_hw.services.tasks.generate_embedding.


embedding_service: EmbeddingService = EmbeddingService()

import json
import math

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

    @staticmethod
    def nearest_predictions(prediction: Prediction, limit: int = 3) -> list:
        current = _vector(getattr(prediction.embedding_values, "embeddings", None))
        if current is None:
            return []

        from data5580_hw.services.database.database_client import db
        from data5580_hw.services.database.prediction_sql import PredictionSQL

        ranked = []
        for row in db.session.query(PredictionSQL).all():
            if row.id == prediction.id or row.embedding is None:
                continue
            other = _vector(_json_value(row.embedding.embedding))
            similarity = _cosine_similarity(current, other)
            if similarity is None:
                continue
            ranked.append({
                "id": row.id,
                "label": float(row.label_numeric) if row.label_numeric is not None else row.label_str,
                "features": _json_value(row.features),
                "similarity": similarity,
            })
        ranked.sort(key=lambda item: item["similarity"], reverse=True)
        return ranked[:limit]


def _vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _cosine_similarity(left, right):
    if left is None or right is None:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


def _json_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


embedding_service: EmbeddingService = EmbeddingService()

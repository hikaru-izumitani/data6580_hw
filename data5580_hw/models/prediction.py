from dataclasses import field, asdict
from dataclasses_json import dataclass_json
import enum
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Union

from data5580_hw.services.database.prediction_sql import EmbeddingSQL, ModelSQL, PredictionSQL, ShapSQL
from data5580_hw.services.database.database_client import db

import json
from decimal import Decimal

def get_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Embedding:
    id: str = field(default_factory=get_id)
    embeddings: Optional[List] = None

    def to_sql(self, prediction_id):
        values = self.embeddings.tolist() if hasattr(self.embeddings, "tolist") else self.embeddings
        return EmbeddingSQL(
            id=self.id,
            prediction_id=prediction_id,
            embedding=json.dumps(values),
        )

    @classmethod
    def from_sql(cls, embedding_sql: EmbeddingSQL) -> 'Embedding':
        return cls(
            id=embedding_sql.id,
            embeddings=json.loads(embedding_sql.embedding) if embedding_sql.embedding else None,
        )


@dataclass
class Shap:
    id: str = field(default_factory=get_id)
    label: Union[None, str] = None
    shap_values: dict = field(default_factory=dict)

    def to_sql(self, prediction_id: str) -> ShapSQL:
        return ShapSQL(
            id=self.id,
            prediction_id=prediction_id,
            label=self.label,
            shap_values=json.dumps(self.shap_values),
        )

    @classmethod
    def from_sql(cls, shap_sql: ShapSQL) -> 'Shap':
        return cls(
            id=shap_sql.id,
            label=shap_sql.label,
            shap_values=json.loads(shap_sql.shap_values),
        )

@dataclass
class Model(object):
    name: str
    version: str
    type: str = None
    threshold: Optional[float] = None
    updated: Optional[datetime] = field(default_factory=datetime.now)
    created: Optional[datetime] = field(default_factory=datetime.now)
    id: str = field(default_factory=get_id)

    # Loaded by the gateway. Not stored, and not part of the API payload.
    _model = None
    _explainer = None
    _embedding = None

    def to_sql(self) -> ModelSQL:
        return ModelSQL(
            id=self.id,
            model_name=self.name,
            model_version=self.version,
            created=self.created,
            updated=self.updated
        )

    @classmethod
    def from_sql(cls, model_sql: ModelSQL)->'Model':
        return cls(
            name=model_sql.model_name,
            version=model_sql.model_version,
            id=model_sql.id,
            #type=model_sql.type,
            updated=model_sql.updated,
            created=model_sql.created
        )


@dataclass
class Prediction(object):
    id: str = field(default_factory=get_id)
    #type: Optional[enum.StrEnum] = None
    model: Optional[Model] = None
    features: Dict = field(default_factory=dict)
    score: Optional[float] = None
    threshold: Optional[float] = None
    label_str: Optional[enum.StrEnum] = None
    label_numeric: Optional[int|float] = None
    actual_str: Optional[enum.StrEnum] = None
    actual_numeric: Optional[int|float] = None
    shap_values: List[Shap] = field(default_factory=list)
    embedding_values: Optional[Embedding] = None
    updated: Optional[datetime] = field(default_factory=datetime.now)
    created: Optional[datetime] = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        data = asdict(self)
        if isinstance(data.get('features'), str):
            try:
                data['features'] = json.loads(data['features'])
            except json.JSONDecodeError:
                pass
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)  # or int, if needed
            return obj

        data = convert_types(data)
        data["label"] = data["label_str"] if data.get("label_str") is not None else data.get("label_numeric")
        raw_shap = data.get("shap_values") or []
        if isinstance(raw_shap, list) and raw_shap:
            data["shap_values"] = raw_shap[0].get("shap_values", {})
        elif not isinstance(raw_shap, dict):
            data["shap_values"] = {}
        
        raw_embedding = data.pop("embedding_values", None) or {}
        coordinates = raw_embedding.get("embeddings") if isinstance(raw_embedding, dict) else None
        if coordinates and len(coordinates) >= 3:
            data["embeddings"] = {
                "kind": "inputs",
                "x": float(coordinates[0]),
                "y": float(coordinates[1]),
                "z": float(coordinates[2]),
            }
        else:
            data["embeddings"] = None
        return data

    def to_prediction_sql(self) -> 'PredictionSQL':
        shap_rows = [item.to_sql(self.id) for item in self.shap_values] if self.shap_values else []
        embedding_sql = self.embedding_values.to_sql(self.id) if self.embedding_values else None
        return PredictionSQL(
            id=self.id,
            #type=self.type,
            model_id=self.model.id,
            features=json.dumps(self.features),
            score=self.score,
            threshold=self.threshold,
            label_str=self.label_str,
            label_numeric=self.label_numeric,
            actual_str=self.actual_str,
            actual_numeric=self.actual_numeric,
            shaps=shap_rows,
            embedding=embedding_sql,
            updated=self.updated,
            created=self.created,
        )

    @classmethod
    def from_prediction_sql(cls, id_:str) -> 'Prediction':
        prediction_sql = db.session.query(PredictionSQL).filter(PredictionSQL.id==id_).first()
        model_sql =db.session.query(ModelSQL).filter(ModelSQL.id==prediction_sql.model_id).first()
        model = Model.from_sql(model_sql)
        shap_values = [Shap.from_sql(shap_sql) for shap_sql in prediction_sql.shaps] if prediction_sql.shaps else []
        embedding_values = Embedding.from_sql(prediction_sql.embedding) if prediction_sql.embedding else None
        return cls(
            id=prediction_sql.id,
            #type=prediction_sql.type,
            model=model,
            features=prediction_sql.features,
            score=prediction_sql.score,
            threshold=prediction_sql.threshold,
            label_str=prediction_sql.label_str,
            label_numeric=prediction_sql.label_numeric,
            actual_str=prediction_sql.actual_str,
            actual_numeric=prediction_sql.actual_numeric,
            shap_values=shap_values,
            embedding_values=embedding_values,
            updated=prediction_sql.updated,
            created=prediction_sql.created,
        )

    @classmethod
    def generate_test_record(cls, *args, **kwargs) -> 'Prediction':
        import faker
        import random

        fake = faker.Faker()

        features = {
            "feature_1": random.randint(0, 100),
            "feature_2": random.randint(-5, 5),
        }
        return cls(
            features=features
        )

    #26-homework-5
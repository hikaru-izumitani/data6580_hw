from dataclasses import field, asdict
from dataclasses_json import dataclass_json
import enum
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Union, List

from data5580_hw.services.database.prediction_sql import PredictionSQL, ModelSQL, ShapSQL, EmbeddingSQL
from data5580_hw.services.database.database_client import db

import json
from decimal import Decimal

def get_id() -> str:
    return uuid.uuid4().hex

@dataclass_json
@dataclass
class Shap:
    id:str = field(default_factory=get_id)
    label: Union[None, str] = None
    shap_values: dict = field(default_factory=dict)

    def to_sql(self, prediction_id):
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

@dataclass_json
@dataclass
class Embedding:
    id: str = field(default_factory=get_id)
    kind: str = "inputs"
    values: list = field(default_factory=list)

    def to_sql(self, prediction_id):
        return EmbeddingSQL(
            id=self.id,
            prediction_id=prediction_id,
            kind=self.kind,
            values=json.dumps(self.values),
        )

    @classmethod
    def from_sql(cls, embedding_sql: EmbeddingSQL) -> 'Embedding':
        return cls(
            id=embedding_sql.id,
            kind=embedding_sql.kind,
            values=json.loads(embedding_sql.values),
        )


def _format_embeddings(embeddings):
    if not embeddings:
        return {}
    embedding = embeddings[0]
    axes = ("x", "y", "z")
    formatted = {"kind": embedding.get("kind") or "inputs"}
    for index, value in enumerate(embedding.get("values") or []):
        key = axes[index] if index < len(axes) else f"d{index}"
        formatted[key] = value
    return formatted


@dataclass_json
@dataclass
class Model(object):
    name: str
    version: str
    type: Optional[str] = None
    threshold: Optional[float] = None
    updated: Optional[datetime] = field(default_factory=datetime.now)
    created: Optional[datetime] = field(default_factory=datetime.now)
    id: str = field(default_factory=get_id)

    _model = None
    _explainer = None
    _umap = None

    def to_sql(self) -> ModelSQL:
        return ModelSQL(
            id=self.id,
            model_name=self.name,
            model_type=self.type,
            model_version=self.version,
            created=self.created,
            updated=self.updated,
        )

    @classmethod
    def from_sql(cls, model_sql: ModelSQL)->'Model':
        return cls(
            name=model_sql.model_name,
            version=model_sql.model_version,
            id=model_sql.id,
            type=model_sql.model_type,
            updated=model_sql.updated,
            created=model_sql.created,
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
    embeddings: List[Embedding] = field(default_factory=list)
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
                return float(obj)  # または必要に応じて int など
            return obj

        data = convert_types(data)
        data["label"] = data["label_str"] if data.get("label_str") is not None else data.get("label_numeric")
        if data.get("shap_values"):
            data["shap_values"] = data["shap_values"][0]["shap_values"]
        else:
            data["shap_values"] = {}
        data["embeddings"] = _format_embeddings(data.get("embeddings"))
        return data

    def to_prediction_sql(self) -> 'PredictionSQL':
        shap_sql = None
        if self.shap_values:
            shap_sql = [shap.to_sql(self.id) for shap in self.shap_values]
        embedding_sql = None
        if self.embeddings:
            embedding_sql = [embedding.to_sql(self.id) for embedding in self.embeddings]

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
            shaps=shap_sql or [],
            embeddings=embedding_sql or [],
            updated=self.updated,
            created=self.created,
        )

    @classmethod
    def from_prediction_sql(cls, id_:str) -> 'Prediction':
        
        prediction_sql = db.session.query(PredictionSQL).filter(PredictionSQL.id==id_).first()
        model_sql =db.session.query(ModelSQL).filter(ModelSQL.id==prediction_sql.model_id).first()
        model = Model.from_sql(model_sql)
        shap_values = None
        if prediction_sql.shaps:
            shap_values = [Shap.from_sql(shap_sql) for shap_sql in prediction_sql.shaps]
        embeddings = None
        if prediction_sql.embeddings:
            embeddings = [Embedding.from_sql(embedding_sql) for embedding_sql in prediction_sql.embeddings]

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
            embeddings=embeddings,
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
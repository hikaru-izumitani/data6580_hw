from dataclasses import field, asdict
from dataclasses_json import dataclass_json
import enum
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

from data5580_hw.services.database.prediction_sql import PredictionSQL, ModelSQL
from data5580_hw.services.database.database_client import db

import json

def get_id() -> str:
    return uuid.uuid4().hex

@dataclass
class Model(object):
    name: str
    version: str
    type: str = None
    updated: Optional[datetime] = field(default_factory=datetime.now)
    created: Optional[datetime] = field(default_factory=datetime.now)
    id: str = field(default_factory=get_id)

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
            type=model_sql.type,
            updated=model_sql.updated,
            created=model_sql.created
        )


@dataclass
class prediction(object):
    id: str = field(default_factory=get_id)
    #type: Optional[enum.StrEnum] = None
    _model: Optional[Model] = None
    features: Dict = field(default_factory=dict)
    score: Optional[float] = None
    threshold: Optional[float] = None
    label_str: Optional[enum.StrEnum] = None
    label_numeric: Optional[int|float] = None
    actual_str: Optional[enum.StrEnum] = None
    actual_numeric: Optional[int|float] = None
    updated: Optional[datetime] = field(default_factory=datetime.now)
    created: Optional[datetime] = field(default_factory=datetime.now)

def to_dict(self) -> dict:
    return asdict(self)

def to_prediction_sql(self) -> 'PredictionSQL':
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
        updated=self.updated,
        created=self.created,
    )

@classmethod
def from_prediction_sql(cls, id_:str) -> 'Prediction':
    prediction_sql = db.session.query(PredictionSQL).filter(PredictionSQL.id==id_).first()
    model_sql =db.session.query(ModelSQL).filter(id=prediction_sql.model_id).first()
    model = Model.from_sql(model_sql)
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
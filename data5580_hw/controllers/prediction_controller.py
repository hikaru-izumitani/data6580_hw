import uuid
from dataclasses import asdict

from flask import jsonify, request

from data5580_hw.models.prediction import Prediction
from data5580_hw.services.database.database_client import db
from data5580_hw.services.prediction_service import prediction_service
#from sqlalchemy.exc import IntegrityError
import json

from data5580_hw.gateways import mlflow_gateway as mlflow

#gateway




class PredictionController(object):
    @staticmethod
    def create_prediction(model_name, model_version, features) -> tuple[dict, int]:
        model = mlflow
        prediction = Prediction(
            features=features,
        )
        prediction = prediction_service.create_service(prediction=prediction)
        return json.dumps(prediction.to_dict()), 200
    def get_prediction(prediction_id: str) -> tuple[dict, int]:
        prediction = Prediction.from_prediction_sql(prediction_id)
        return json.dumps(prediction.to_dict()), 200
    def get_prediction_sql(cls, id_: str) -> 'PredictionSQL':
        return db.session.query(PredictionSQL).filter_by(id=id_).first()
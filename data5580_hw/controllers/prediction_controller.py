import uuid
from dataclasses import asdict

from flask import jsonify, request

from data5580_hw.models.prediction import Prediction
from data5580_hw.services.database.database_client import db
from data5580_hw.services.prediction_service import prediction_service
from data5580_hw.services.database.prediction_sql import PredictionSQL

#from sqlalchemy.exc import IntegrityError
import json

from data5580_hw.gateways import mlflow_gateway

#gateway




class PredictionController(object):
    @staticmethod
    def create_prediction(model_name, model_version, features) -> tuple[dict, int]:
        #model = mlflow
        try:
            prediction = Prediction(
                features=features,
            )
            #prediction = prediction_service.create_service(prediction=prediction)
            prediction = prediction_service.create_prediction(
                model_name=model_name,
                model_version=model_version,
                prediction=prediction)
            return jsonify(prediction.to_dict()), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except (KeyError, FileNotFoundError, LookupError) as e:
            error_msg = f"Model '{model_name}' (version '{model_version}') was not found"
            return jsonify({"error": error_msg}), 404
        except Exception as e:
            return jsonify({"error": "An internal error occurred."}), 500
    def get_prediction(prediction_id: str) -> tuple[dict, int]:
        prediction = Prediction.from_prediction_sql(prediction_id)
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
        return jsonify(prediction.to_dict()), 200
    @classmethod
    def get_prediction_sql(cls, id_: str) -> 'PredictionSQL':
        return db.session.query(PredictionSQL).filter_by(id=id_).first()

prediction_controller = PredictionController()
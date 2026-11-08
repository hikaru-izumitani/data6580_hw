from asyncio.log import logger
from typing import Dict
import logging

from sqlalchemy.exc import IntegrityError

import pandas as pd

from data5580_hw.gateways import mlflow_gateway
from data5580_hw.models.prediction import Prediction, Model
from data5580_hw.services.database.prediction_sql import PredictionSQL
from data5580_hw.services.inference_service import inference_service
from data5580_hw.services.database.database_client import db

logger = logging.getLogger(__name__)


class PredictionService(object):
    def __init__app__(self,app):
        ...
    def create_prediction(self, model_name: str, model_version: str, prediction: Prediction) -> Prediction:
        #model = Model(
            #name=model_name,
            #version=model_version
        #)
        model = mlflow_gateway.mlflow_gateway.get_model(model_name, model_version)
        # prediction.model = model
        # inference_service calls prediction.model._model. The gateway stores the model on model.model.
        model._model = model.model
        prediction.model = model

        values = list(prediction.features.values()) if isinstance(prediction.features, dict) else list(prediction.features)
        # expected = model.model.n_features_in_
        expected = model._model.n_features_in_

        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("Invalid input data: features must be numeric.")
        if len(values) != expected:
            raise ValueError(f"Invalid input data: expected {expected} features, got {len(values)}.")

        # Previously predicted here. Model execution now goes through create_inference.
        # features_df = pd.DataFrame([prediction.features])
        # prediction.score = float(model.model.predict(features_df)[0])
        # prediction.label_numeric = prediction.score
        original_features = prediction.features
        prediction.features = pd.DataFrame([original_features])
        prediction = inference_service.create_inference(prediction)
        prediction.features = original_features
        prediction.label_numeric = float(prediction.label_numeric[0])
        prediction.score = prediction.label_numeric

        model_sql = model.to_sql()
        prediction_sql = prediction.to_prediction_sql()
        try:
            db.session.add_all([model_sql, prediction_sql])
            db.session.commit()
        except IntegrityError as e:
            logger.error(str(e))
            raise e
        prediction = Prediction.from_prediction_sql(prediction_sql.id)
        #user_sql = db.session.query(PredictionSQL).filter_by(id=prediction.id).first()
        logger.info(f"Created prediction: {prediction.id} for {prediction.model.name} {prediction.model.version}."
        f"features={original_features} label={prediction.label_numeric}")
        return prediction

prediction_service = PredictionService()
from asyncio.log import logger
from typing import Dict
import logging

from sqlalchemy.exc import IntegrityError

from data5580_hw.models.prediction import Prediction, Model
from data5580_hw.services.database.prediction_sql import PredictionSQL
from data5580_hw.services.inference_service import inference_service
from data5580_hw.services.database.database_client import db

logger = logging.getLogger(__name__)


class PredictionService(object):
    def __init__app__(self,app):
        ...
    def create_prediction(self, prediction: Prediction) -> Prediction:
        model = Model(
            name='test',
            version='1'
        )

        prediction.model = model

        prediction = inference_service.create_inference(prediction)

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
        logger.info(f"Created prediction: {prediction.id} for {prediction.model_name} {prediction.model.version}.")
        return prediction

prediction_service = PredictionService()
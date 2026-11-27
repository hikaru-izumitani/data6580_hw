import uuid
from dataclasses import asdict

from flask import jsonify, request

from data5580_hw.models.prediction import Prediction, TaskStatus
from data5580_hw.services.database.database_client import db
from data5580_hw.services.prediction_service import prediction_service
from data5580_hw.services.database.prediction_sql import PredictionSQL

#from sqlalchemy.exc import IntegrityError
import json

from data5580_hw.gateways import mlflow_gateway
from data5580_hw.gateways.arize_gateway import arize_gateway, save_for_later

#gateway




class PredictionController(object):
    @staticmethod
    def create_prediction(model_name, model_version) -> tuple[dict, int]:
        #model = mlflow
        try:
            request_data = request.get_json(silent=True)
            if not isinstance(request_data, dict) or "features" not in request_data:
                raise ValueError("Invalid input data: features are required.")
            features = request_data["features"]
            prediction = Prediction(
                features=features,
            )
            #prediction = prediction_service.create_service(prediction=prediction)
            # prediction = prediction_service.create_prediction(
            #     model_name=model_name,
            #     model_version=model_version,
            #     prediction=prediction)
            block = request_data.get("block", True)
            if isinstance(block, str):
                block = block.strip().lower() not in {"false", "0", "no"}
            prediction = prediction_service.create_prediction(
                model_name=model_name,
                model_version=model_version,
                prediction=prediction,
                block=bool(block),
            )
            # return jsonify(prediction.to_dict()), 200
            body = prediction.to_dict()
            if prediction.label_numeric is not None:
                body["label"] = float(prediction.label_numeric)
            tasks = getattr(prediction, "tasks", None)
            if tasks:
                body["tasks"] = tasks
            return jsonify(body), 200 if block else 202
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except (KeyError, FileNotFoundError, LookupError) as e:
            error_msg = f"Model '{model_name}' (version '{model_version}') was not found"
            return jsonify({"error": error_msg}), 404
        except Exception as e:
            return jsonify({"error": "An internal error occurred."}), 500
    @staticmethod
    def record_actual(prediction_id):
        body = request.get_json(silent=True) or {}
        if "actual" not in body:
            return jsonify({"error": "actual is required"}), 400
        row = db.session.query(PredictionSQL).filter_by(id=prediction_id).first()
        if row is None:
            return jsonify({"error": "Prediction not found"}), 404
        row.actual_numeric = float(body["actual"])
        db.session.commit()
        prediction = Prediction.from_prediction_sql(prediction_id)
        try:
            arize_gateway.send_actual(prediction)
        except Exception as error:
            save_for_later("actual", prediction_id, str(error))
            return jsonify({"id": prediction_id, "actual": float(body["actual"])}), 200
        return jsonify({"id": prediction_id, "actual": float(body["actual"])}), 200
    # def get_prediction(prediction_id: str) -> tuple[dict, int]:
    def get_prediction(self, prediction_id: str) -> tuple[dict, int]:
        prediction = Prediction.from_prediction_sql(prediction_id)
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
        return jsonify(prediction.to_dict()), 200

    @staticmethod
    def get_task_status(task_id: str):
        from flask import current_app

        # result = current_app.extensions["celery"].AsyncResult(task_id)
        # error = str(result.result) if result.state == "FAILURE" else None
        # task_result = result.result if result.state == "SUCCESS" else None
        # status = TaskStatus(
        #     task_id=task_id,
        #     state=result.state,
        #     result=None if task_result is None else str(task_result),
        #     error=error,
        # )
        # return jsonify(status.to_dict()), 200
        task = current_app.extensions["celery"].AsyncResult(task_id)
        if task.state == "PENDING":
            status = TaskStatus(task_id=task_id, state="PENDING", status="Pending...")
            return jsonify(status.to_dict()), 200
        elif task.state == "STARTED":
            status = TaskStatus(task_id=task_id, state="STARTED", status="Started...")
            return jsonify(status.to_dict()), 200
        elif task.state == "PROGRESS":
            status = TaskStatus(task_id=task_id, state="PROGRESS", progress=task.info)
            return jsonify(status.to_dict()), 200
        elif task.state == "SUCCESS":
            status = TaskStatus(task_id=task_id, state="SUCCESS", result=str(task.info))
            return jsonify(status.to_dict()), 200
        elif task.state == "FAILURE":
            status = TaskStatus(task_id=task_id, state="FAILURE", error=str(task.info))
            return jsonify(status.to_dict()), 500
        else:
            # RETRY
            status = TaskStatus(task_id=task_id, state=task.state, error=str(task.info))
            return jsonify(status.to_dict()), 500
    @classmethod
    def get_prediction_sql(cls, id_: str) -> 'PredictionSQL':
        return db.session.query(PredictionSQL).filter_by(id=id_).first()

prediction_controller = PredictionController()
from flask import Blueprint#, request

# In-app modules
from data5580_hw.controllers.prediction_controller import prediction_controller

prediction_blueprint = Blueprint('prediction', __name__)


@prediction_blueprint.route('/v1/<model_name>/version/<model_version>/predict', methods=['POST'])
def create_prediction(model_name, model_version):

    #request_data = request.get_json()
    #features = request_data["features"]
    return prediction_controller.create_prediction(model_name, model_version)#, features)


@prediction_blueprint.route('/prediction/<id>', methods=['GET'])
def get_user(id: str):
    return prediction_controller.get_prediction(id)

@prediction_blueprint.route('/prediction/<prediction_id>/actual', methods=['POST'])
def record_actual(prediction_id):
    return prediction_controller.record_actual(prediction_id)
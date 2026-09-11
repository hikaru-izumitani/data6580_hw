# Stdlib
import uuid

# Pip install section
from flask import Blueprint, Flask, jsonify, request

# In-app modules
from data5580_hw.models.user_model import User

user_blueprint = Blueprint('user',__name__, url_prefix='/user')

@user_blueprint.route('/create', method=['POST'])
def create_user():
    request_data = request.get_json()

    id_ = uuid.uuid4().hex



    # except KeyError as e
    try:
        user_ = User(
            id = id_,
            name = request_data['name'],
            email = request_data['email']
        )
    except KeyError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(user_), 200
import uuid

from flask import jsonify, request

from data5580_hw.models.user_model import User


class UserController(object):

    @staticmethod
    def create_user() -> tuple[str, int]:

        request_data = request.get_json()

        id_ = uuid.uuid4().hex

        try:
            user_ = User(
                id=id_,
                name=request_data['name'],
                email=request_data['email'],
            )


        except KeyError as e:
            return jsonify({'error': str(e)}), 400

        return jsonify(user_), 200


user_controller = UserController()

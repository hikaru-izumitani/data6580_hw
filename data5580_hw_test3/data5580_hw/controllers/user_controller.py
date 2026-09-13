import uuid
from dataclasses import asdict

from flask import jsonify, request

from data5580_hw.models.user_model import User
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.user_sql import UserSQL
from sqlalchemy.exc import IntegrityError

import re

# check re
EMAIL_REGEX = r'^[\w\.-]+@[\w\.-]+\.\w+$'


def validate_email(email):
  if not email or not isinstance(email, str):
    return False
  return bool(re.match(EMAIL_REGEX, email))

class UserController(object):

    @staticmethod
    def create_user() -> tuple[str, int]:

        request_data = request.get_json()
        if not validate_email(request_data['email']):
            return jsonify({'error': 'Invalid email format'}), 400

        id_ = uuid.uuid4().hex

        try:
            user_ = User(
                id=id_,
                name=request_data['name'],
                email=request_data['email'],
            )

            user_sql = user_.to_user_sql()

            db.session.add(user_sql)

            db.session.commit()

            user_sql = db.session.query(UserSQL).filter(UserSQL.id == id_).one_or_none()

            user_ = User.from_user_sql(user_sql)

        except KeyError as e:
            return jsonify({'error': str(e)}), 400

        except IntegrityError:
            db.session.rollback()
            return jsonify({'error': 'the email is already in use'}),400
        return jsonify(asdict(user_)), 200
        
        """return (
            jsonify({
                'message': 'User created successfully.',
                'user': asdict(user_), 
            }),
            200,
        )"""

    def get_user(self, user_id: str) -> tuple[str, int]:

        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).one_or_none()

        if not user_sql:
            return jsonify({"error": "User not found"}),404

        user_ = User.from_user_sql(user_sql)

        return jsonify(asdict(user_)), 200

    def patch_user(self, user_id:str) -> tuple[str, int]:
        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).one_or_none()
        if not user_sql:
            return jsonify({"error": "User not found"}),404

        request_data = request.get_json()
        if not request_data or ('name' not in request_data and 'email' not in request_data):
            return jsonify({'error':'No valid fields to update. You need name or email to update'}), 400
        if 'name' in request_data:
            user_sql.name = request_data['name']
        if 'email' in request_data:
            if not validate_email(request_data['email']):
                return jsonify({'error': 'Invalid email format'}), 400
            user_sql.email = request_data['email']
        db.session.commit()
        user_ = User.from_user_sql(user_sql)
        return jsonify(asdict(user_)), 200

    def delete_user(self, user_id:str) -> tuple[str, int]:
        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).one_or_none()
        if not user_sql:
            return jsonify({"error": "User not found"}), 404
        db.session.delete(user_sql)
        db.session.commit()
        return jsonify({'message':'User deleted successfully'}), 200

    def fetch_all_users(self) -> tuple[str, int]:
        users = db.session.query(UserSQL).all()

        user_list = [asdict(User.from_user_sql(user)) for user in users]

        return jsonify(user_list), 200

    def put_user(self, user_id:str)->tuple[str, int]:
        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).one_or_none()
        if not user_sql:
            return jsonify({"error": "User not found"}),404

        request_data = request.get_json()
        if not request_data:
            return jsonify({'error':'No valid fields to update. You need name and email.'}), 400
        if 'name' not in request_data:
            return jsonify({'error':'You need name to update.'}), 400
        if 'email' not in request_data:
            return jsonify({'error':'You need email to update'}), 400
        if not validate_email(request_data['email']):
            return jsonify({'error': 'Invalid email format'}), 400

        user_sql.name = request_data['name']
        user_sql.email = request_data['email']

        db.session.commit()
        user_ = User.from_user_sql(user_sql)
        return (
            jsonify({
                'message': 'User details updated successfully.',
                'user': asdict(user_), 
            }),
            200,
        )



        

        




user_controller = UserController()

"""File 2 controllers/user_controller.py for File 1 app.py"""
from flask import Blueprint, jsonify, request
from services.user_service import UserService

user_bp = Blueprint("user", __name__, url_prefix="/users")

# サービス層のインスタンスを用意
user_service = UserService()

# HW3 acceptance Criteria
"""
1. As a user, I want to create a new user, so that I can store my information in the database.
Acceptance Criteria:
The system should accept a POST request at /users with required user details (e.g., name, email).
If the user already exists, return an error with a message indicating duplication.
If the creation is successful, return a success message with the user s ID and details.
The ID should be a nuance
"""
@user_bp.route("", methods=["POST"])
def create_user():
  """endpoint to make users"""
  request_data = request.get_json() or {}

  name = request_data.get("name")
  email = request_data.get("email")

  if not name or not email:
    return jsonify({"error": "Name and email are required"}),400

  try:
    # ビジネスロジック（サービス層）に処理を任せる
    user_obj = user_service.create_user(name=name, email=email)
    #(
        #name=request_data.get("name"), email=request_data.get("email")
    #)
    return jsonify(user_obj.to_dict()), 201
  except ValueError as e:
    return jsonify({"error": str(e)}), 400
  except Exception as e:
    return jsonify({"error": str(e)}), 500 #400


@user_bp.route("", methods=["GET"])
def get_all_users():
  """すべてのユーザーを取得するエンドポイント"""
  users = user_service.get_all_users()
  return jsonify([user.to_dict() for user in users]), 200
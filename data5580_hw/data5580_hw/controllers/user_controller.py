from flask import Blueprint, jsonify, request
from services.user_service import UserService

user_bp = Blueprint("user", __name__, url_prefix="/users")

# サービス層のインスタンスを用意
user_service = UserService()


@user_bp.route("/create", methods=["POST"])
def create_user():
  """ユーザーを作成するエンドポイント"""
  request_data = request.get_json()

  try:
    # ビジネスロジック（サービス層）に処理を任せる
    user_obj = user_service.create_user(
        name=request_data.get("name"), email=request_data.get("email")
    )
    return jsonify(user_obj.to_dict()), 201
  except KeyError as e:
    return jsonify({"error": f"Missing required field: {e}"}), 400
  except Exception as e:
    return jsonify({"error": str(e)}), 400


@user_bp.route("/", methods=["GET"])
def get_all_users():
  """すべてのユーザーを取得するエンドポイント"""
  users = user_service.get_all_users()
  return jsonify([user.to_dict() for user in users]), 200
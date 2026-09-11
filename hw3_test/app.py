import logging
from flask import Flask, jsonify, request
from prometheus_flask_exporter import PrometheusMetrics

from adapters.memory_repository import InMemoryUserRepository
from use_cases.user_service import UserService

# 1. 標準ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # 標準出力へ出力
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 2. Prometheus 監視の初期化 (/metrics エンドポイントが自動生成される)
metrics = PrometheusMetrics(app)

# 依存性の注入 (DI)
repo = InMemoryUserRepository()
users = repo.users
user_service = UserService(repo)

@app.route("/users", methods=["POST"])
def create_user():
    req_data = request.get_json(silent=True)
    if not req_data or "name" not in req_data or "email" not in req_data:
        logger.warning("Create user failed: Missing required fields")
        return jsonify({"error": "Missing required fields: name and email"}), 400

    new_user, error = user_service.create_user(req_data.get("name"), req_data.get("email"))
    if error:
        logger.warning(f"Create user failed: {error}")
        return jsonify({"error": error}), 400

    logger.info(f"User created successfully: ID {new_user['id']}")
    return jsonify({"message": "User created successfully", "user": new_user}), 201

@app.route("/users", methods=["GET"])
def get_all_users():
    logger.info("Fetching all users")
    return jsonify(user_service.list_users()), 200

@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = user_service.get_user(user_id)
    if not user:
        logger.warning(f"User not found: ID {user_id}")
        return jsonify({"error": "The user was not found"}), 404
    logger.info(f"Fetched user: ID {user_id}")
    return jsonify({"user": user}), 200

@app.route("/users/<int:user_id>", methods=["PATCH"])
def update_user(user_id):
    req_data = request.get_json(silent=True)
    if not req_data:
        logger.warning("Update user (PATCH) failed: Request body is missing")
        return jsonify({"error": "Request body is missing"}), 400
    if "name" not in req_data and "email" not in req_data:
        logger.warning("Update user (PATCH) failed: No fields provided")
        return jsonify({"error": "At least one field (name or email) must be provided for update"}), 400

    updated_user, error = user_service.update_user(
        user_id, 
        name=req_data.get("name"), 
        email=req_data.get("email")
    )
    if error:
        logger.warning(f"Update user (PATCH) failed for ID {user_id}: {error}")
        return jsonify({"error": error}), 400
    if not updated_user:
        logger.warning(f"User not found for update (PATCH): ID {user_id}")
        return jsonify({"message": "The user was not found"}), 404

    logger.info(f"User updated successfully (PATCH): ID {user_id}")
    return jsonify({"message": "User updated successfully", "user": updated_user}), 200

@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user_put(user_id):
    req_data = request.get_json(silent=True)
    if not req_data:
        logger.warning("Update user (PUT) failed: Request body is empty")
        return jsonify({"error": "Request body is empty"}), 400
    
    missing_fields = [field for field in ["name", "email"] if field not in req_data]
    if missing_fields:
        logger.warning(f"Update user (PUT) failed: Missing fields {missing_fields}")
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    updated_user, error = user_service.update_user(
        user_id, 
        name=req_data.get("name"), 
        email=req_data.get("email")
    )
    if error:
        logger.warning(f"Update user (PUT) failed for ID {user_id}: {error}")
        return jsonify({"error": error}), 400
    if not updated_user:
        logger.warning(f"User not found for update (PUT): ID {user_id}")
        return jsonify({"message": "The user was not found"}), 404

    logger.info(f"User details updated successfully (PUT): ID {user_id}")
    return jsonify({"message": "User details updated successfully", "user": updated_user}), 200

@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    success = user_service.delete_user(user_id)
    if not success:
        logger.warning(f"Delete user failed: User not found ID {user_id}")
        return jsonify({"message": "The user was not found"}), 404
    
    logger.info(f"User deleted successfully: ID {user_id}")
    return jsonify({"message": "User deleted successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
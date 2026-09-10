from flask import Flask, jsonify
from services.database.database_client import db
from controllers.user_controller import user_bp

def create_app():
    app = Flask(__name__)

    # データベースの接続設定（必要に応じてSQLiteやPostgreSQLのURIを指定）
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 1. データベースの初期化
    db.init_app(app)

    # 2. データベースのテーブルを作成
    with app.app_context():
        db.create_all()

    # 3. コントローラー（Blueprint）の登録
    app.register_blueprint(user_bp)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Hello, Flask Clean Architecture!"})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

"""from flask import Flask, jsonify, request"""

# $body = @{
#     a = 5
#     b = 7
# } | ConvertTo-Json
#
# Invoke-RestMethod `
#     -Uri "http://localhost:5000/add" `
#     -Method Post `
#     -ContentType "application/json" `
#     -Body $body
"""
def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Hello, Flask!"})

    @app.route("/add", methods=["POST"])
    def add():
        data = request.get_json()
        a = data.get("a")
        b = data.get("b")

        if a is None or b is None:
            return jsonify({"error": "Missing values"}), 400

        return jsonify({"result": a + b})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
"""
import uuid
from gateways.user_repository import SQLAlchemyUserRepository


class UserService:
  """ユーザーに関するビジネスロジックを管理するサービスクラス"""

  def __init__(self):
    self.repo = SQLAlchemyUserRepository()

  def create_user(self, name: str, email: str):
    """ユーザーを作成するためのビジネスロジック"""
    # ここでIDの生成や、必要に応じたバリデーション（入力値チェック）を行う
    user_id = uuid.uuid4().hex
    return self.repo.create(user_id=user_id, name=name, email=email)

  def get_all_users(self):
    """すべてのユーザーを取得するビジネスロジック"""
    return self.repo.find_all()
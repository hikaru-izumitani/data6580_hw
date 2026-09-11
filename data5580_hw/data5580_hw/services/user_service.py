"""File 3 services/user_service.py for File 2 controllers/user_controller.py"""
import uuid
from gateways.user_repository import SQLAlchemyUserRepository


class UserService:
  """ユーザーに関するビジネスロジックを管理するサービスクラス"""

  def __init__(self):
    self.repo = SQLAlchemyUserRepository()

  def create_user(self, name: str, email: str):
    """Business Logic to make users"""
    existing_user = self.repo.find_by_email(email)
    if existing_user:
      raise ValueError(f"User with email '{email}' already")


    # ここでIDの生成や、必要に応じたバリデーション（入力値チェック）を行う
    user_id = uuid.uuid4().hex
    return self.repo.create(user_id=user_id, name=name, email=email)

  def get_all_users(self):
    """business logic to get all users"""
    return self.repo.find_all()
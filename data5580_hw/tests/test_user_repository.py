# tests/test_user_repository.py
from datetime import datetime
import pytest
from data5580_hw.gateways.user_repository import SQLAlchemyUserRepository
from data5580_hw.models.user_model import UserSQL
from data5580_hw.services.database.database_client import db
from flask import Flask


@pytest.fixture
def app():
  """テスト用のFlaskアプリとインメモリDBをセットアップするフィクスチャ"""
  app = Flask(__name__)
  app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
  app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

  db.init_app(app)

  with app.app_context():
    db.create_all()  # テーブルを作成
    yield app
    db.session.remove()
    db.drop_all()  # テスト終了後にクリーンアップ


@pytest.fixture
def repo():
  return SQLAlchemyUserRepository()


def test_create_and_find_user(app, repo):
  with app.app_context():
    # 1. ユーザー作成のテスト
    user_id = "user_123"
    name = "Hikaru"
    email = "hikaru@example.com"

    created_user = repo.create(user_id, name, email)
    assert created_user.id == user_id
    assert created_user.name == name
    assert created_user.email == email

    # 2. ID検索のテスト
    found_user = repo.find_by_id(user_id)
    assert found_user is not None
    assert found_user.name == "Hikaru"
    assert found_user.email == "hikaru@example.com"
import pytest
from data5580_hw.app import create_app

@pytest.fixture
def app():
    """テスト用の Flask アプリケーション・インスタンスを生成するフィクスチャ"""
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    
    with app.app_context():
        yield app
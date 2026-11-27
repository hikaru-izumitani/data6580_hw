import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data5580_hw.config import Config
import data5580_hw.gateways.mlflow_gateway as gw

# GitHub Actions has no .env. The tests mock the Groq call, so a placeholder
# is enough for the client to be constructed. A real key in the environment
# or in data5580_hw/.env is left as-is.
os.environ.setdefault("GROQ_API_KEY", "test-key")

# Pytest has no Redis. Keep it off the database the running app uses.
_test_db = Path(tempfile.mkdtemp()) / "test.db"
Config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{_test_db}"

# create_app() loads the model from localhost:8080.
# Tests replace that load with a fake model first.
_fake_model = MagicMock()
_fake_model.predict.return_value = [1.5]
_fake_model.n_features_in_ = 8
_fake_model.feature_names_in_ = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
_fake_explainer = MagicMock()
_fake_explainer.predict.return_value = [[0.4, -0.1, 0.05, -0.02, 0.01, 0.03, -0.2, 0.08]]

_fake_embedding = MagicMock()
_fake_embedding.predict.return_value = [[0.24, -0.38, 0.4]]

gw.MLFlowGateway._load_models = lambda self, model_uri, flavor: _fake_model
gw.MLFlowGateway._load_explainer = lambda self, explainer_path: _fake_explainer
gw.MLFlowGateway._load_embedding = lambda self, embedding_path: _fake_embedding

from data5580_hw.app import create_app

@pytest.fixture
def app():
    """FOR TESTING FIXTURE"""
    app = create_app()
    # The workflow runs pytest without Redis or a Celery worker.
    # Eager tasks run inside this process when delay() is called.
    celery_app = app.extensions["celery"]
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    app.config.update({
        "TESTING": True,
    })

    with app.app_context():
        yield app
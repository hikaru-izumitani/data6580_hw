from unittest.mock import MagicMock

import pytest

import data5580_hw.gateways.mlflow_gateway as gw

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
gw.MLFlowGateway._load_models = lambda self, model_uri, flavor: _fake_model
gw.MLFlowGateway._load_explainer = lambda self, explainer_path: _fake_explainer

from data5580_hw.app import create_app

@pytest.fixture
def app():
    """FOR TESTING FIXTURE"""
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    
    with app.app_context():
        yield app
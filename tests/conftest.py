from unittest.mock import MagicMock

import pytest

import data5580_hw.gateways.mlflow_gateway as gw

# create_app() loads the model from localhost:8080.
# Tests replace that load with a fake model first.
_fake_model = MagicMock()
_fake_model.predict.return_value = [1.5]
_fake_model.n_features_in_ = 8
gw.MLFlowGateway._load_models = lambda self, model_uri, flavor: _fake_model

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
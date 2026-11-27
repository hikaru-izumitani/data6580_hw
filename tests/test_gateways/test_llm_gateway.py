import pytest

from data5580_hw.gateways.llm_gateway import llm_gateway
from data5580_hw.models.prediction import Prediction, Shap


def test_generate_explanation_sends_prediction_shap_and_neighbors(app, monkeypatch):
    captured = {}

    async def fake_explain(agent, prediction, shap_values, similar_observations):
        captured["prediction"] = prediction
        captured["shap_values"] = shap_values
        captured["similar_observations"] = similar_observations
        return "The nearest block is priced higher because income differs."

    monkeypatch.setattr(
        "data5580_hw.gateways.llm_gateway.embedding_service.nearest_predictions",
        lambda prediction: [{"id": "near", "label": 2.0, "similarity": 1.0}],
    )
    monkeypatch.setattr(llm_gateway, "explain_prediction", fake_explain)

    prediction = Prediction(
        id="current",
        features={"MedInc": 8.3},
        label_numeric=1.5,
        score=1.5,
        shap_values=[Shap(shap_values={"MedInc": 0.4})],
    )

    summary = llm_gateway.generate_explanation(prediction)

    assert summary == "The nearest block is priced higher because income differs."
    assert captured["prediction"]["id"] == "current"
    assert captured["prediction"]["label"] == 1.5
    assert captured["prediction"]["features"] == {"MedInc": 8.3}
    assert captured["shap_values"] == {"MedInc": 0.4}
    assert captured["similar_observations"] == [{"id": "near", "label": 2.0, "similarity": 1.0}]


def test_generate_explanation_returns_a_friendly_error_when_the_call_fails(app, monkeypatch):
    async def fail(agent, prediction, shap_values, similar_observations):
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "data5580_hw.gateways.llm_gateway.embedding_service.nearest_predictions",
        lambda prediction: [],
    )
    monkeypatch.setattr(llm_gateway, "explain_prediction", fail)

    with pytest.raises(RuntimeError, match="The explanation could not be generated"):
        llm_gateway.generate_explanation(Prediction(id="current", label_numeric=1.5))

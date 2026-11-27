from data5580_hw.models.prediction import Embedding, Model, Prediction
from data5580_hw.services.database.database_client import db
from data5580_hw.services.embedding_service import embedding_service


def _store(vector, label):
    model = Model(name="california-housing", version="5", type="REGRESSION")
    prediction = Prediction(
        model=model,
        features={"MedInc": label},
        label_numeric=label,
        embedding_values=Embedding(embeddings=vector),
    )
    db.session.add(model.to_sql())
    db.session.add(prediction.to_prediction_sql())
    db.session.commit()
    return prediction


def test_nearest_predictions_prefers_the_same_direction(app):
    current = _store([1, 0, 0], 1.0)
    same_direction = _store([2, 0, 0], 2.0)
    perpendicular = _store([0, 1, 0], 3.0)
    _store([0, 0, 0], 4.0)

    nearest = embedding_service.nearest_predictions(current)

    assert [item["id"] for item in nearest] == [same_direction.id, perpendicular.id]
    assert nearest[0]["similarity"] == 1.0
    assert nearest[0]["label"] == 2.0
    assert nearest[1]["similarity"] == 0.0
    assert current.id not in {item["id"] for item in nearest}


def test_nearest_predictions_is_empty_without_an_embedding(app):
    prediction = Prediction(
        model=Model(name="california-housing", version="5", type="REGRESSION"),
        features={"MedInc": 1.0},
        label_numeric=1.0,
    )

    assert embedding_service.nearest_predictions(prediction) == []

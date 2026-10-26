# HW6: Add SHAP Values to Predictions

This app predicts one house price with a California Housing decision tree, then splits that prediction into a contribution from each feature (SHAP). The explanation is the breakdown of the single input you sent, not a summary of the whole dataset.

The model and the explainer are not in Git. `*.db` is ignored, so a first run has nothing registered. Train and register the model in MLflow before starting the app.

## 1. Environment

Run these commands from the project root.

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train and register the model

Use two terminals. Leave MLflow running in the first one.

```zsh
source .venv/bin/activate
mlflow server --port 8080 --backend-store-uri sqlite:///mlruns.db
```

In the second terminal, train the model and register both the prediction model and the SHAP explainer in the Model Registry. The first run needs a network connection to download the California Housing data.

```zsh
source .venv/bin/activate
python generate_regression_model.py
```

On an empty registry, the prediction model `california-housing` is version **1**, and the explainer `california-housing-explainer` is also version **1**. If models are already registered, the prediction model gets the next version number. Check these lines in the log:

```text
Created version '1' of model 'california-housing'.
Set explainer_path to california-housing-explainer/1
```

Set `MODELS` in `data5580_hw/config.py` to those version numbers. The values in this repository, version `4` and `california-housing-explainer/1`, match the registry used during development. On a new registry, use the numbers from the log, for example:

```python
MODELS = {
    'california-housing': {
        "1": {
            "model": None,
            "model_type": "REGRESSION",
            "mlflow_flavor": "sklearn",
            "explainer_path": "california-housing-explainer/1",
            ...
        }
    }
}
```

`"1"` is the prediction model version. `explainer_path` is the string printed by the script. If these do not match the registry, the app cannot load the model.

## 3. Start the app

Keep MLflow running, and run this in another terminal. `PYTHONPATH` is not required.

```zsh
source .venv/bin/activate
python -m data5580_hw.app
```

At startup the app loads `models:/california-housing/<version>` and `models:/california-housing-explainer/<version>`. Flask listens on `http://127.0.0.1:5000`.

On macOS, `localhost:5000` can be taken by AirPlay and return an empty response. Send requests to `127.0.0.1`.

## 4. Send a prediction

Use the same feature names and column order as the California Housing training data.

```zsh
curl -s -X POST http://127.0.0.1:5000/model/california-housing/version/1 \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "MedInc": 8.3252,
      "HouseAge": 41.0,
      "AveRooms": 6.984127,
      "AveBedrms": 1.02381,
      "Population": 322.0,
      "AveOccup": 2.555556,
      "Latitude": 37.88,
      "Longitude": -122.23
    }
  }'
```

Replace `version/1` in the URL with the prediction model version in `config.py`.

Read a stored result with the `id` from the response. The app recreates its SQLite database on startup, so an `id` from before a restart is gone.

```zsh
curl -s http://127.0.0.1:5000/prediction/paste-the-id-here
```

## 5. How to read the response

The regression target is the median house value for a block group, in units of $100,000. A `label` of `3.822` means about $382,000. For this model, `label`, `label_numeric`, and `score` are the same number. `label_str` is for classification, so it is `null` for this regression model.

| Field | Meaning |
|---|---|
| `label` | The prediction for this one input, in units of $100,000 |
| `shap_values` | How far each feature moved that prediction from the model's baseline |
| `features` | The input you sent |
| `model.name` / `model.version` | The registered model that was used |
| `model.type` | `REGRESSION` |

A positive SHAP value is the amount that feature pushed the prediction above the model's average prediction. A negative value is the amount it pushed the prediction below that average. The unit is the same as `label`: $100,000. A larger absolute value means a stronger effect on this prediction.

| Feature | Meaning |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

For the sample input above, `MedInc` is a large positive value and `Latitude` is negative. High income pushed the price up, and latitude pushed it down.

A POST to a version that does not exist returns `404` and `Model 'california-housing' (version '...') was not found`. If SHAP computation fails, the API returns `400` and puts the reason in `{"error": "..."}`.

## 6. Tests

Start MLflow, then align the version in `config.py` and in the test with the registered version. The tests in this repository request version `4`.

```zsh
pytest tests/test_services/test_prediction_service.py
```

# HW7: Add UMAP Embeddings to Predictions

This app predicts one house price, then maps that same input into a 3D UMAP embedding. The embedding is stored with the prediction and returned next to `label`. The fitted model is not in Git, so the registry starts empty.

## 1. Environment

Run these commands from the project root.

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train and register the models

Use two terminals. Leave MLflow running in the first one.

```zsh
source .venv/bin/activate
mlflow server --port 8080 --backend-store-uri sqlite:///mlruns.db
```

In the second terminal, train the decision tree and register the prediction model, the SHAP explainer, and the UMAP model. The first run needs a network connection to download the California Housing data.

```zsh
source .venv/bin/activate
python generate_regression_model.py
```

On an empty registry, all three registered names are version **1**. Confirm these lines in the log:

```text
Created version '1' of model 'california-housing'.
Set explainer_path to california-housing-explainer/1
Set umap_path to california-housing-umap/1
```

`data5580_hw/config.py` already uses those three paths. If the log prints a different version, put that number in `MODELS` before starting the app. `"1"` is the prediction model. `explainer_path` and `umap_path` are separate models.

UMAP is fit once, with the parameters at the top of `generate_regression_model.py`:

```python
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3
```

Change those constants and run the script again to register a new UMAP version. Then set `umap_path` to the version printed in the log. The app does not refit UMAP on each request.

## 3. Start the app

Keep MLflow running.

```zsh
source .venv/bin/activate
python -m data5580_hw.app
```

Startup downloads three artifact sets: the housing model (`5` files), the explainer (`8` files), and the UMAP model (`8` files). Debug mode loads them a second time after `Restarting with stat`. Requests do not download them again. Flask listens on `http://127.0.0.1:5000`.

On macOS, `localhost:5000` can be taken by AirPlay and return an empty response. Send requests to `127.0.0.1`.

## 4. Send a prediction

Use the same feature names as the California Housing training data.

```zsh
curl -s -X POST http://127.0.0.1:5000/v1/california-housing/version/1/predict \
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

`version/1` must match the prediction-model key in `config.py`. A successful response has `label`, `shap_values`, and `embeddings` together. The embedding block looks like this:

```json
"embeddings": {
  "kind": "inputs",
  "x": 7.67,
  "y": 0.56,
  "z": 16.17
}
```

`kind` is `"inputs"` because the coordinates describe the features you sent, not the predicted price. `x`, `y`, and `z` are the three UMAP components. The numbers above are an example of the shape. A newly trained UMAP model will not return these exact coordinates.

`label` is the house-price prediction in dollars. For the version `1` model this repository was developed against, the sample input returns `468800`. `label_numeric` and `score` stay in the training unit of $100,000, so they are `4.688` for that same model. `DecisionTreeRegressor` has no fixed `random_state`, so a new training run can return a different price. The embedding coordinates are not dollars.

`actual_numeric` and `actual_str` are `null`. The request does not include the true price.

## 5. Check that the embedding was stored

Copy `id` from the response. The app recreates its SQLite database on startup, so an `id` from before a restart is gone.

```zsh
curl -s http://127.0.0.1:5000/prediction/paste-the-id-here
```

The stored record returns the same `embeddings` object, including `kind`, `x`, `y`, and `z`.

## 6. Check that the UMAP model stays fixed

Send the same `curl` a second time. `id` changes, because this is a new prediction row. `embeddings.x`, `embeddings.y`, and `embeddings.z` stay the same. That is the persisted UMAP model: it was fit during registration and is only transformed at request time.

## 7. Check the error response

A UMAP failure is returned as `400` with the reason in `{"error": "..."}`. The automated test forces that failure. It does not happen for this valid sample input, so `curl` cannot show it.

```zsh
pytest tests/test_services/test_prediction_service.py::test_umap_failure_returns_descriptive_error
```

The expected body is:

```json
{"error": "UMAP computation failed: invalid input data"}
```

MLflow must still be running. The test loads the models named in `config.py`.

## 8. Run the prediction tests

```zsh
pytest tests/test_services/test_prediction_service.py
```

`test_create_prediction` checks the sample prediction, the dollar `label`, the SHAP signs, a 3-number embedding with `kind` `"inputs"`, and that a second prediction returns the same coordinates. It expects `label_numeric` of about `4.688` and `label` of `468800`. If your newly trained tree predicts a different price, change those two expected numbers to the `label_numeric` and `label` your model returns. The embedding checks do not depend on that price.

## What this covers

| Requirement | What you should see |
|---|---|
| Accept the usual feature input | The POST body is only `features` |
| Compute a 3D UMAP embedding after the prediction | `embeddings` has `x`, `y`, and `z` |
| Make `n_neighbors`, `min_dist`, and `n_components` configurable | The three constants in `generate_regression_model.py` |
| Store the embedding with the prediction | `GET /prediction/<id>` returns the same coordinates |
| Return `label` and `embeddings` | Both fields are in the POST response |
| Return a descriptive error if UMAP fails | The `400` test in section 7 |
| Keep one fitted UMAP model across calls | The second POST repeats `x`, `y`, and `z` |

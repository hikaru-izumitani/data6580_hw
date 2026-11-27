# data6580_hw

## Setting
```zsh
git clone -b hw5 https://github.com/hikaru-izumitani/data6580_hw.git
```
Set a virtual environment of Python
```zsh
python -m venv .venv
```

Then, use the virtual environment of Python
```zsh
source .venv/bin/activate
```
Setup Python packages

```zsh
pip install -r requirements.txt
pip install pylint
```


## Run
```zsh
python -m data5580_hw.app
```

Prediction, SHAP, and the embedding are no longer computed in one request. Redis queues the two background jobs, and a Celery worker runs them. Start each process in its own terminal, from this directory, with the virtual environment activated. MLflow must already be serving the models at `http://localhost:8080`.

Redis is the queue and the place where Celery stores task results. `0` at the end of the URL is the Redis database index, not a port.

```zsh
brew services start redis
redis-cli ping
```

`PONG` means Redis is accepting connections. The broker and result backend both default to `redis://localhost:6379/0`. Override them with `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` in `data5580_hw/.env`.

The worker loads the Flask app on its own, so start it before Flask. Leave this terminal open. It is ready when the log prints `ready.`

```zsh
celery -A data5580_hw.celery_worker.celery worker --loglevel=info
```

Flask only saves the prediction and enqueues the jobs. It does not compute SHAP or the embedding itself.

```zsh
python -m data5580_hw.app
```

`"block": false` returns immediately with `202`. `shap_values` is empty and `embeddings` is `null`, because those jobs have only been queued. The response includes `id` plus `tasks.explanation_task_id` and `tasks.embedding_task_id`.

```zsh
curl -s -X POST http://127.0.0.1:5000/v1/california-housing/version/5/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[8.3252,41.0,6.984127,1.02381,322.0,2.555556,37.88,-122.23],"block":false}'
```

Task status does not contain the SHAP values. It only tells you whether that job has finished. Repeat this until `state` is `SUCCESS`. `PENDING` means the worker has not picked it up yet.

```zsh
curl -s http://127.0.0.1:5000/tasks/<task_id>
```

After both tasks are `SUCCESS`, read the prediction again with the `id` from the first response. That record now has `shap_values` and `embeddings`.

```zsh
curl -s http://127.0.0.1:5000/prediction/<prediction_id>
```

`"block": true` is the default. Flask waits for both jobs and returns `200` with `shap_values` and `embeddings` in that same response, so there is nothing to poll.

```zsh
curl -s -X POST http://127.0.0.1:5000/v1/california-housing/version/5/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[8.3252,41.0,6.984127,1.02381,322.0,2.555556,37.88,-122.23],"block":true}'
```

`python -m data5580_hw.app` listens on port 5000. On this Mac that port already returned `403` before Flask started, so start Flask on 5050 and use 5050 in every curl below. If 5000 is free, keep the commands above and replace `5050` with `5000`.

```zsh
python -c "from data5580_hw.app import create_app; create_app().run(debug=True, port=5050)"
```

The explanation route calls Groq. Copy `data5580_hw/.env_example` to `data5580_hw/.env`, set `GROQ_API_KEY`, and do not commit `.env`. The example file lists every variable the app reads. `LLM_PROVIDER` is optional and defaults to `groq`.

```zsh
cp data5580_hw/.env_example data5580_hw/.env
```

`LLM_PROVIDER` defaults to `groq`. The route waits for Groq and adds `summary` to the same prediction JSON. Call it only after `shap_values` and `embeddings` are present, which `"block": true` already guarantees.

```zsh
curl -s http://127.0.0.1:5050/prediction/<prediction_id>/explainer
```

One stored prediction has no neighbor, because the search skips itself. `summary` then ends with `No relevant predictions were found for comparison.` Create a second prediction, then ask for the explanation of the first id again. The summary should name the other prediction and the differences in price and features.

```zsh
curl -s -X POST http://127.0.0.1:5050/v1/california-housing/version/5/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[2.0,20.0,4.0,1.0,1000.0,3.0,34.0,-118.0],"block":true}'
```

```zsh
curl -s http://127.0.0.1:5050/prediction/<first_prediction_id>/explainer
```

Starting Flask drops the SQLite database, so predictions from the previous run are gone. Groq sometimes rejects one call and accepts the next. The failed response is `The explanation could not be generated. Please try again.` Run the same curl again. Wait a few seconds between explanation calls so the free-tier limit is less likely to trip.

If you want to see the depencency of the package
```zsh
pyreverse -o png data5580_hw
```
it generates package.png and classes.png
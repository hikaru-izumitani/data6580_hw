# Explainer を一つの処理に戻して見る

HW6 の explainer は、最初から複数ファイルに分かれていました。hexagonal architecture では、SHAP の計算、MLflow からの読み込み、SQLite への保存、HTTP の応答を同じ関数に置かないためです。意味を追うときは、分かれる前の一本に戻すと分かります。下のコードは、今の部品をその順に戻したものです。このファイル自体はアプリからは呼ばれません。

## 構築する順番

内側の「何を計算して、何を保存するか」から作り、外側の HTTP は最後につなぎます。route と `app.py` は予測 API のときから既にあるので、HW6 では中を足しています。

1. `models/prediction.py` と `services/database/prediction_sql.py`。`Shap` と `shaps` テーブル。返したい `label` と `shap_values` の形を先に決める。
2. `generate_regression_model.py`。決定木と `shap.Explainer` を作り、MLflow に登録する。アプリの外のスクリプト。
3. `config.py`。登録された version と `explainer_path` を書く。
4. `gateways/mlflow_gateway.py`。起動時にその住所から予測モデルと explainer を読む。
5. `services/explainer_service.py`。配列を特徴量名の辞書にする。
6. `services/prediction_service.py`。価格を予測したあと、その辞書を作って保存する。
7. `controllers/prediction_controller.py`。成功を JSON にし、失敗を 400 や 404 のメッセージにする。
8. `routes/prediction_routes.py`。`POST /model/<name>/version/<version>` で `features` を受け取る。ここは元からある。
9. `app.py`。`config.py` を読み、route を登録し、`gateway.init_app` で explainer を載せる。ここも元からある。

## 一つの場合

各行の `#` は、その処理が今入っているファイルです。このスクリプト自体はアプリからは呼ばれません。

```python
import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeRegressor

# generate_regression_model.py
# アプリの外。決定木を学習し、MLflow の california-housing として登録する。
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# generate_regression_model.py の SHAPModel
# shap.Explainer を pickle し、california-housing-explainer として登録する。
explainer = shap.Explainer(model)

# routes/prediction_routes.py
# POST の JSON から features を取り、controller に渡す。
features = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}

# config.py
# "4" が予測モデルの version、explainer_path が explainer の住所。
model_version = "4"
explainer_path = "california-housing-explainer/1"

# app.py
# 起動時に config を読み、routes を登録し、gateway.init_app を呼ぶ。
# gateways/mlflow_gateway.py
# models:/california-housing/4 と models:/california-housing-explainer/1 を読む。
loaded_model = model
loaded_explainer = explainer

# services/prediction_service.py
# 先に価格を予測する。回帰ではこの値が label になる。
features_df = pd.DataFrame([features])
label = float(loaded_model.predict(features_df)[0])

# services/explainer_service.py
# 予測のあと。配列を、特徴量名から寄与への辞書にする。
raw = loaded_explainer.shap_values(features_df)
values = np.asarray(raw).reshape(-1)
shap_values = {
    name: float(value)
    for name, value in zip(features_df.columns, values)
}

# models/prediction.py
# Shap として shaps テーブルへ保存し、to_dict で応答の形にする。
response = {
    "label": label,                 # 3.822。単位は 10 万ドル
    "shap_values": shap_values,     # {"MedInc": 2.045, "Latitude": -0.405, ...}
}

# controllers/prediction_controller.py
# 成功なら response を 200 で返す。
# SHAP の失敗は 400、config に無い version は 404。
print(response)
```

ここでやっていることは 4 つだけです。

1. 決定木が、この 1 件の価格 `label` を出す。
2. `shap.Explainer(model)` が、その木に合った説明器を作る。
3. `shap_values` が、特徴量の数だけ数値の配列を返す。
4. 列名と並べて、`MedInc` のような名前付きの辞書にする。

プラスはその特徴量が予測を平均より上げた分、マイナスは下げた分です。単位は `label` と同じです。

## 最初にサービスへ切り出したとき

リポジトリに最初に入った `ExplainerService` は、上の 3 と 4 だけを関数にしたものです。配列を名前付き辞書にせず、`predict` の戻り値をそのまま `Shap` に入れていました。

```python
class ExplainerService:
    @staticmethod
    def create_explanation(prediction: Prediction) -> Prediction:
        shap_values_ = prediction.model._explainer.predict(prediction.features)

        prediction.shap_values = [
            Shap(label=None, shap_values=shap_values_)
        ]
        return prediction
```

`prediction.model._explainer` は、すでにどこかで読み込まれている前提です。学習と読み込みはこの関数の外です。この時点では、応答が `{"MedInc": 2.045}` にならず、配列のまま保存されていました。

## 今どのファイルに分かれているか

| 上の一本の中の処理 | 今のファイル | 層の役割 |
|---|---|---|
| `DecisionTreeRegressor` を学習する | `generate_regression_model.py` | アプリの外。モデルを作って MLflow に登録する |
| `shap.Explainer(model)` を作り、pickle して登録する | 同ファイルの `SHAPModel` | explainer を `models:/california-housing-explainer/<version>` として保存する |
| 起動時にその explainer を読む | `gateways/mlflow_gateway.py` | 外の MLflow とアプリの間。`model._explainer` に載せる |
| 読む先の名前 | `config.py` の `explainer_path` | `california-housing-explainer/1` のような Registry の住所 |
| DataFrame にして `predict` し、名前付き辞書にする | `services/explainer_service.py` | 説明の計算だけ。保存も HTTP もしない |
| 予測のあとにその関数を呼ぶ | `services/prediction_service.py` | 1 件の予測という仕事の順番 |
| 辞書を `shaps` テーブルへ書く | `models/prediction.py` と `services/database/prediction_sql.py` | ドメインの `Shap` と、SQLite の行 |
| `label` とフラットな `shap_values` で返す | `models/prediction.py` の `to_dict` | API の形 |
| `features` を受け取り、失敗を 400 や 404 にする | `controllers/prediction_controller.py` | HTTP とアプリの間 |
| URL から `features` を取り出す | `routes/prediction_routes.py` | どの URL が controller を呼ぶか |
| 起動時に config、route、gateway をつなぐ | `app.py` | プロセスの入口 |

`explainer_service.py` に残っているのは、上の一本のうちこの部分だけです。

```python
features_df = pd.DataFrame([features])
raw = prediction.model._explainer.predict(features_df)
values = np.asarray(raw).reshape(-1)
shap_values = {name: float(value) for name, value in zip(features_df.columns, values)}
```

`predict` になっているのは、MLflow の pyfunc が `shap_values` を `SHAPModel.predict` の中で呼んでいるためです。アプリ側は explainer を直接触らず、`models:/...` の `predict` を呼びます。

## なぜ分けたか

一つの関数に置くと、その関数が MLflow、SHAP、SQLite、Flask の全部を知ることになります。hexagonal architecture では、説明の計算は「1 行の特徴量を受け取り、名前付きの寄与を返す」だけにします。モデルの保存場所が MLflow であること、結果を SQLite に書くこと、HTTP が JSON であることは、外側の gateway と controller が担当します。explainer の中身を `TreeExplainer` から別の説明器に変えても、予測を保存するテーブルと URL はそのままにできます。

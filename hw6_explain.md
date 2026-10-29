# HW6 で変えたコードの意味

HW6 は、既存の Flask + MLflow の予測 API に SHAP を足したものです。予測値を計算したあと、その 1 件について「どの特徴量がどれだけ予測を動かしたか」を計算し、予測と同じレコードとして SQLite に保存し、API の応答に入れます。

リクエスト 1 件は、次の順でコードを通ります。

1. `prediction_controller.py` が `features` を受け取る
2. `prediction_service.py` が MLflow からモデルを取り、住宅価格を予測する
3. `explainer_service.py` が SHAP を特徴量名付きの辞書にする
4. `prediction.py` と `prediction_sql.py` がその辞書を `shaps` テーブルへ書く
5. `prediction.py` の `to_dict` が `label` と `shap_values` を応答にする

## `generate_regression_model.py`

予測モデルとは別に、SHAP を計算する explainer を Model Registry へ登録します。アプリはリクエストのたびに `shap.Explainer` を作らず、ここで保存したものを起動時に読みます。

`SHAPModel` は MLflow の pyfunc です。`load_context` が pickle した explainer を開き、`predict` が `shap_values` を返します。中に入っているのは `shap.Explainer(model)` で、この課題のモデルは決定木なので、木の構造を使った SHAP になります。

```python
registered_model_name=REGISTERED_EXPLAINER_NAME
```

この引数で、run の中だけでなく Registry の `california-housing-explainer` としても登録されます。アプリが読む URI は `models:/california-housing-explainer/1` のような形です。スクリプト末尾の `Set explainer_path to ...` が、`config.py` に書く文字列です。

空の Registry では予測モデルも explainer も version 1 です。このリポジトリの `config.py` が `"4"` なのは、開発中の Registry に既に version 1〜3 があったためです。

## `data5580_hw/config.py`

アプリがどの登録モデルを読むかを書いています。

- `"4"` は予測モデル `california-housing` の version です。Gateway は `models:/california-housing/4` を読みます。
- `explainer_path` は SHAP 用モデルの名前と version です。Gateway は先頭に `models:/` を付けます。
- `model_type: "REGRESSION"` は、このモデルが数値を予測することを示します。分類ラベルは使いません。

`threshold` の `APPROVE` / `DECLINE` は分類用の下書きです。今回の回帰の予測経路では使っていません。

## `data5580_hw/gateways/mlflow_gateway.py`

起動時に、予測モデルと explainer をメモリへ載せる場所です。

```python
explainer_path = model_config.get("explainer_path")
if explainer_path:
    self.models[model_name][version]["explainer"] = mlflow.pyfunc.load_model(
        f"models:/{explainer_path}"
    )
else:
    self.models[model_name][version]["explainer"] = None
```

`explainer_path` が空文字や未設定のときは読みにいきません。以前の `is not None` だと、空文字でも `models:/` を読もうとして起動に失敗していました。

`get_model` では、読み込んだ explainer を `model._explainer` に付けます。`model_.get("explainer")` なので、キーが無くてもここで落ちません。`_explainer` はデータベースには保存しません。MLflow 上のモデル本体への参照です。

## `data5580_hw/services/database/prediction_sql.py`

SHAP を保存するテーブルと、予測との関係です。

`ShapSQL` の 1 行は、1 件の予測に属する SHAP です。

- `prediction_id` は、どの予測の説明かを指す外部キーです。
- `shap_values` は、特徴量名と寄与の辞書を JSON 文字列として入れた CLOB です。
- `label` は分類でクラスごとに説明を分けるための列です。この回帰では `null` です。

`PredictionSQL.shaps` は、1 件の予測が複数の SHAP 行を持てる関係です。`ShapSQL.prediction` はその逆で、1 行の SHAP が 1 件の予測に属します。`back_populates` は相手の属性名と一致している必要があります。こちらが `shaps` なので、相手は `prediction` です。

`cascade="all, delete-orphan"` は、予測を `db.session.add` したとき、その予測に入っている SHAP 行も同じ commit で書き込むという意味です。

`ModelSQL.predictions` は別の関係です。1 つのモデルに予測が複数ある、という既存のつながりで、SHAP 用の `prediction` とは名前が違っていて問題ありません。

## `data5580_hw/models/prediction.py`

データベースの行と、API が返す辞書を行き来する場所です。

`Shap` は 1 件分の説明です。`to_sql` は自分の `id` と、辞書を `json.dumps` した文字列を `ShapSQL` にします。`from_sql` はそれを `json.loads` で辞書に戻します。

`Prediction.shap_values` の初期値は `field(default_factory=list)` です。`typing.List` を初期値にすると、`Prediction(...)` を作った時点で失敗します。

`to_prediction_sql` は、説明があるとき各 `Shap` を `ShapSQL` にし、`PredictionSQL` の `shaps` に渡します。引数名はクラスの属性名 `shaps` と揃えます。説明が無いときは `[]` です。`None` を渡すと relationship が受け取りません。

`from_prediction_sql` は、保存後にその予測を読み直します。`prediction_sql.shaps` が空でなければ、各行を `Shap.from_sql(shap_sql)` で戻します。`if` は、行があるときだけリストを作るためのものです。

`to_dict` が API の形を作ります。

```python
data["label"] = data["label_str"] if data.get("label_str") is not None else data.get("label_numeric")
if data.get("shap_values"):
    data["shap_values"] = data["shap_values"][0]["shap_values"]
```

内部では `shap_values` は `Shap` のリストです。応答では、その中の辞書を一段上げて `{"MedInc": 2.045, ...}` にします。回帰の `label` は `label_numeric` です。分類なら `label_str` を使います。

`Prediction` から `@dataclass_json` を外しています。このデコレータはクラスに書いた `to_dict` をライブラリの `to_dict` で上書きするため、外さないと `label` が無く、`shap_values` がネストしたまま返っていました。

`Model.to_sql` の列名は `model_type` です。`type=` という引数は `ModelSQL` に無いので、保存時に失敗します。

## `data5580_hw/services/explainer_service.py`

SHAP の配列を、特徴量名と寄与の辞書に変える場所です。

入力が辞書なら、そのキーを列名にした 1 行の DataFrame にします。リストなら、学習時にモデルが覚えた `feature_names_in_` を列名にします。列名が無いと、応答の `shap_values` が `0` や `1` になり、どの特徴量の寄与か分かりません。

`prediction.model._explainer.predict(features_df)` が、登録済み explainer の計算です。戻り値は 1 行 × 特徴量数の配列なので、`reshape(-1)` で 1 列に伸ばし、列名と `zip` して `float` の辞書にします。

explainer が無いとき、計算が失敗したとき、個数が列数と合わないときは `ValueError` です。コントローラはこの例外を `400` と `{"error": "メッセージ"}` にします。

## `data5580_hw/services/prediction_service.py`

1 件の予測の順番を決めています。

1. Gateway からモデルを 1 回取る
2. `prediction.model` に付ける
3. `model.model.predict` で価格を予測し、`label_numeric` と `score` に入れる
4. そのあと `explainer_service.create_explanation` を呼ぶ
5. モデル、予測、SHAP をまとめて commit する
6. 保存した行を読み直し、そのオブジェクトを返す

説明の計算は、モデルを付けて予測したあとにあります。先に呼ぶと、まだ `_explainer` が付いていないオブジェクトを見ることになります。

この回帰経路では `inference_service` は呼んでいません。Gateway がセットするのは `model.model` で、`inference_service` が読むのは `model._model` だからです。

## `data5580_hw/controllers/prediction_controller.py`

HTTP の入口です。`create_prediction` は `features` だけの `Prediction` を作り、モデルの取得はサービスに任せます。

コメントにしてある `mlflow_gateway.get_model` は、import したモジュール側には無く、インスタンス側にあります。`try` の外で呼ぶと、サービスに入る前に落ちます。

`ValueError` は `400` で、例外メッセージをそのまま返します。SHAP の失敗はここに入ります。`KeyError` はモデル名や version が `config.py` に無いときで、`404` と `Model '...' (version '...') was not found` を返します。

`get_prediction` の `@staticmethod` は、GET `/prediction/<id>` の `id` をこの引数へ渡すためのものです。これが無いと、URL の `id` ではなくコントローラ自身が引数になります。

## `data5580_hw/services/inference_service.py`

今回の回帰リクエストでは使っていません。モデル種別で予測の仕方を分ける下書きです。

`arize` の `ModelTypes` は、今入っている版には無いので import を外しています。比較は `"REGRESSION"` と `"SCORE_CATEGORICAL"` という文字列です。`config.py` の `model_type` と同じ文字です。

## テストと依存関係

`tests/test_services/test_prediction_service.py` は 3 つを確認します。

- version `4` で `label` が約 `3.822` になり、8 個の SHAP があり、`MedInc` がプラス、`Latitude` がマイナスであること
- 無い version が `404` と決まった文を返すこと
- SHAP が失敗したとき `400` と、その失敗理由を返すこと

`requirements.txt` に足した `arize`、`notebook`、`matplotlib` は、分類の下書きと SHAP の学習ノート用です。予測 API の実行に直接必要なのは、元からある `shap`、`mlflow`、`scikit-learn`、`pandas` です。

`data5580_hw/notebooks/shap_regression_binary_multiclass_walkthrough.ipynb` は、回帰・二値分類・多クラスで SHAP の形がどう変わるかを見るノートです。API からは呼ばれません。

`.gitignore` の `memo.txt` は、課題文をリポジトリに含めないための指定です。

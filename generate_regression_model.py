# mlflow server --port 8080 --backend-store-uri sqlite:///mlruns.db

import mlflow
from tempfile import TemporaryDirectory
import pickle
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
import shap
import umap
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Configuration
# ----------------------------
EXPERIMENT_NAME = "california-housing"
REGISTERED_MODEL_NAME = "california-housing"
REGISTERED_EXPLAINER_NAME = "california-housing-explainer"

REGISTERED_EMBEDDING_NAME = "california-housing-umap"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------
# Load dataset
# ----------------------------
housing = fetch_california_housing()

# Create features X and target y.
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # Median house value in $100,000s

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define a custom pyfunc model
class SHAPModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load model and SHAP explainer
        with open(context.artifacts["model"], 'rb') as fin:
            self.model = pickle.load(fin)

    def predict(self, context, model_input, params=None):
        # Calculate SHAP values
        return self.model.shap_values(model_input)


class EmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the fitted embedding model
        with open(context.artifacts["model"], 'rb') as fin:
            self.model = pickle.load(fin)

    def predict(self, context, model_input, params=None):
        # Project inputs onto the fitted embedding
        return self.model.transform(model_input)

# ----------------------------
# Train + Log
# ----------------------------
with mlflow.start_run() as run:

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model_type", "DecisionTreeRegressor")
    # mlflow.log_param("fit_intercept", model.fit_intercept)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log model with signature
    signature = infer_signature(X_train, model.predict(X_train))

    logged_model = mlflow.sklearn.log_model(
        sk_model=model,
        #name="model",
        artifact_path="model",
        signature=signature,
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    run_id = run.info.run_id

    print(logged_model)

    # log an explainer
    explainer = shap.Explainer(model)

    with TemporaryDirectory() as temp_dir:
        local_path = temp_dir + '/explainer.pkl'

        with open(local_path, 'wb') as fout:
            pickle.dump(explainer, fout)

        artifacts = {
            "model": local_path
        }

        response = mlflow.pyfunc.log_model(
            artifact_path='model-explainer'
            , python_model=SHAPModel()
            , artifacts=artifacts
            , input_example=X_test.fillna(0).sample(n=10, random_state=123)
            , signature=mlflow.models.signature.infer_signature(model_input=X_test,
                                                                model_output=explainer.shap_values(X_test))
            , extra_pip_requirements=['shap']
            , registered_model_name=REGISTERED_EXPLAINER_NAME
        )

        print(
            f"Registered explainer {REGISTERED_EXPLAINER_NAME} "
            f"version {response.registered_model_version}"
        )
        print(
            "Set explainer_path to "
            f"{REGISTERED_EXPLAINER_NAME}/{response.registered_model_version}"
        )
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        random_state=42,
    )
    reducer.fit(X_train)

    with TemporaryDirectory() as temp_dir:
        local_path = temp_dir + '/embedding.pkl'
        with open(local_path, 'wb') as fout:
            pickle.dump(reducer, fout)
        
        artifacts = {"model": local_path}
        embedded = reducer.transform(X_test)
        response = mlflow.pyfunc.log_model(
            artifact_path='model-embedding',
            python_model=EmbeddingModel(),
            artifacts=artifacts,
            input_example=X_test.fillna(0).sample(n=10, random_state=123),
            signature=mlflow.models.signature.infer_signature(model_input=X_test,
                                                                model_output=embedded),
            extra_pip_requirements=['umap-learn'],
            registered_model_name=REGISTERED_EMBEDDING_NAME
        )

        print(
            "Set embedding_path to "
            f"{REGISTERED_EMBEDDING_NAME}/{response.registered_model_version}"
        )




print(f"Run logged with ID: {run_id}")

import mlflow

from data5580_hw.models.prediction import Model


class MLFlowGateway:
    def __init__(self):
        self.models = {}
    def init_app(self, app):
        mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])

        self.models = app.config["MODELS"]

        #for model in self.models.keys():
            #for version in self.models[model].keys():
                #flavor_ = self.models[model][version].get("flavor", 'pyfunc')
                #self.models[model][version]["model"] = mlflow.pyfunc.load_model(self._get_model_uri(model, version), flavor=flavor_)
                #model_config = self.models[model_name][version]
        for model_name, versions in self.models.items():
            for version, model_config in versions.items():
                # 1. Use 'mlflow_flavor' from config.
                flavor = model_config.get("mlflow_flavor", model_config.get("flavor", "pyfunc"))
                
                uri = self._get_model_uri(model_name, version)
                
                # 2. Load the model with _load_models.
                self.models[model_name][version]["model"] = self._load_models(uri, flavor)

                explainer_path = model_config.get("explainer_path")
                if explainer_path:
                    self.models[model_name][version]["explainer"] = self._load_explainer(explainer_path)
                else:
                    self.models[model_name][version]["explainer"] = None
    def _get_model_uri(self, name, version):
        return f"models:/{name}/{version}"

    def _load_models(self, model_uri, flavor):
        if flavor == 'pyfunc':
            return mlflow.pyfunc.load_model(model_uri)
        elif flavor == 'sklearn':
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError(f"Invalid flavor: {flavor}")

    def _load_explainer(self, explainer_path):
        return mlflow.pyfunc.load_model(f"models:/{explainer_path}")

    def get_model(self, name, version):
        model_ = self.models[name][version]

        model = Model(
            type=model_["model_type"],
            name=name,
            version=str(version),
            threshold=model_["threshold"]
        )

        model.model = model_["model"]
        model._model = model_["model"]
        model._explainer = model_.get("explainer")

        return model

mlflow_gateway = MLFlowGateway()
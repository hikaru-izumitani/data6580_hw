import mlflow

from data5580_hw.models.prediction import Model


class MLFlowGateway:

    models = {}
    def __init_app__(self):
        mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])

        self.models = app.config["MODELS"]

        for model in self.models.keys():
            for version in self.models[model].keys():
                flavor_ = self.models[model][version].get("flavor", 'pyfunc')
                self.models[model][version]["model"] = mlflow.pyfunc.load_model(self._get_model_uri(model, version), flavor=flavor_)
    def _get_model_uri(self, name, version):
        return f"models:/{name}/{version}"

    def _load_models(self, model_uri, flavor):
        if flavor == 'pyfunc':
            return mlflow.pyfunc.load_model(model_uri)
        elif flavor == 'sklearn':
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError(f"Invalid flavor: {flavor}")

        
    def get_model(self, name, version):
        model_ = self.models[name][version]

        model = Model(
            type=model_["model_type"],
            name=name,
            version=model_["version"],
            threshold=model_["threshold"]
        )

        model.model = model_["model"]

        return model

mlflow_gatewa = MLFlowGateway()
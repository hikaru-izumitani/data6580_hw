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
                # 1. Config側の 'mlflow_flavor' に対応させる
                flavor = model_config.get("mlflow_flavor", model_config.get("flavor", "pyfunc"))
                
                uri = self._get_model_uri(model_name, version)
                
                # 2. 定義してある _load_models を使ってロードする
                self.models[model_name][version]["model"] = self._load_models(uri, flavor)
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
            version=str(version),
            threshold=model_["threshold"]
        )

        model.model = model_["model"]

        return model

mlflow_gateway = MLFlowGateway()
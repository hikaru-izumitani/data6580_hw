import logging
import os


class Config:

    # In-file sqlite database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data.db'

    # In memory database
    # SQLALCHEMY_DATABASE_URI = ':memory:///data.db'
    #SQLALCHEMY_ECHO = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    LOGGING_LEVEL = logging.DEBUG
    MLFLOW_TRACKING_URI = "http://localhost:8080"
    TRACKING_URI = "http://localhost:8080"
    MODELS = {
        'california-housing': {
            "4": {
                "model": None,
                "model_type": "REGRESSION",
                "mlflow_flavor": "sklearn",
                "explainer_path": "california-housing-explainer/1",
                "umap_path": "california-housing-umap/1",
                "labels": None,
                "threshold": {
                    "threshold": 0.6,
                    "ABOVE_THRESHOLD": "APPROVE",
                    "BELOW_THRESHOLD": "DECLINE"

                }

            }
    }
    }
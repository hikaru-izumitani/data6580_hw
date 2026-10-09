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

    TRACKING_URI = "http://localhost:8080"
    MODELS = {
        'california-housing': {
            "2": {
                "model": None,
                "model_type": "REGRESSION",
                "mlflow_flavor": "sklearn",
                "labels": None,
                "threshold": None

            }
    }
    }
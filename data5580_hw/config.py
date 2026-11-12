import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


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

    ARIZE_API_KEY = os.getenv('ARIZE_API_KEY')
    SPACE_ID = os.getenv('ARIZE_SPACE_ID')
    ARIZE_ENABLED = os.getenv("ARIZE_ENABLED", "true").lower() == "true"
    ARIZE_ENVIRONMENT = os.getenv("ARIZE_ENVIRONMENT", "production")
    
    MODELS = {
        'california-housing': {
            "5": {
                "model": None,
                "model_type": "REGRESSION",
                "mlflow_flavor": "sklearn",
                "explainer_path": "california-housing-explainer/2",
                "embedding_path": "california-housing-umap/3",
                "labels": None,
                "threshold": None

            }
    }
    }
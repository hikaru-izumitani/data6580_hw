import sys
from pathlib import Path

# Celery adds the project directory only while importing this module, then
# removes it. macOS starts pool workers with spawn, so the path has to stay
# or those processes cannot import data5580_hw.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data5580_hw import app

flask_app = app.create_app(reset_db=False)

celery = flask_app.extensions["celery"]

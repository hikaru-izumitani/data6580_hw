from data5580_hw import app

flask_app = app.create_app()

celery = flask_app.extensions["celery"]

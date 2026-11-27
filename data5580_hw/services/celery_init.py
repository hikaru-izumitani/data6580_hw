import logging

from celery import Celery, Task
from celery.signals import worker_ready, worker_shutdown

logger = logging.getLogger(__name__)


@worker_ready.connect
def log_worker_ready(**kwargs):
    logger.info("Celery worker started")


@worker_shutdown.connect
def log_worker_shutdown(**kwargs):
    logger.info("Celery worker stopped")


def celery_init_app(app):

    class FlaskTask(Task):

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(
        app.import_name,
        task_cls=FlaskTask,
        include=[
            # "data5580_hw.services.embedding_service",
            # "data5580_hw.services.explainer_service",
            "data5580_hw.services.tasks",
        ]
    )

    celery_app.config_from_object({
        "broker_url": app.config.get(
            "CELERY_BROKER_URL",
            "redis://localhost:6379/0"
        ),
        "result_backend": app.config.get(
            "CELERY_RESULT_BACKEND",
            "redis://localhost:6379/0"
        ),
        "task_track_started": True,
        "result_expires": 3600,
    })

    celery_app.set_default()

    app.extensions["celery"] = celery_app

    return celery_app

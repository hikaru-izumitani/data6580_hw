# app.py
from flask import Flask, request, jsonify
from celery import Celery
import time
import os

# celery -A app.celery worker --loglevel=info -P gevent

# Flask app setup
app = Flask(__name__)

# Celery configuration
app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery instance
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Background task
@celery.task(bind=True)
def long_task(self, duration):
    """Simulates a long-running task."""
    for i in range(duration):
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': duration})
    return {'status': 'Task completed!', 'total_seconds': duration}

@app.route('/start-task', methods=['POST'])
def start_task():
    """Start a background task."""
    try:
        duration = int(request.json.get('duration', 5))
        if duration <= 0:
            return jsonify({'error': 'Duration must be positive'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid duration'}), 400

    task = long_task.apply_async(args=[duration])
    return jsonify({'task_id': task.id}), 202

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Check task status."""
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'state': task.state, 'status': 'Pending...'})
    elif task.state == 'PROGRESS':
        return jsonify({'state': task.state, 'progress': task.info})
    elif task.state == 'SUCCESS':
        return jsonify({'state': task.state, 'result': task.info})
    else:
        # FAILURE or RETRY
        return jsonify({'state': task.state, 'error': str(task.info)}), 500

if __name__ == '__main__':
    app.run(debug=True)
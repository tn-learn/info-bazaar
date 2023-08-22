from celery import Celery, Task
import time
from llamapi import REDIS_BROKER_URL, REDIS_BACKEND_URL

# Celery setup using the configurations from __init__.py
celery_app = Celery(__name__, broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL)

# Load Huggingface model and tokenizer here (omitted for brevity)
# They will be loaded once when the worker starts.


@celery_app.task(bind=True, base=Task)
def run_inference(self, texts: list):
    print(
        f"Running inference for task ID: {self.request.id} on worker: {self.request.hostname}"
    )
    # Simulated inference logic
    # Replace with your actual model inference
    time.sleep(10)
    predictions = [text[0:5] for text in texts]
    return predictions

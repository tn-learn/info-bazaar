import guidance
from celery import Celery, Task

from bazaar.lem_utils import get_llm, clean_program_string
from llamapi import REDIS_BROKER_URL, REDIS_BACKEND_URL

# Celery setup using the configurations from __init__.py
celery_app = Celery(__name__, broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL)


@celery_app.task(bind=True, base=Task)
def run_inference(self, program_string: str, inputs: dict, guidance_kwargs: dict):
    print(
        f"Running inference for task ID: {self.request.id} on worker: {self.request.hostname}"
    )
    model_name = guidance_kwargs.pop("model_name")
    program_string = clean_program_string(program_string)
    program = guidance(program_string, llm=get_llm(model_name), **guidance_kwargs)
    outputs = program(**inputs)
    return outputs


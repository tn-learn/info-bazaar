import guidance
from celery import Celery, Task
from typing import List, Dict

from bazaar.lem_utils import get_llm, clean_program_string
from llamapi import REDIS_BROKER_URL, REDIS_BACKEND_URL

# Celery setup using the configurations from __init__.py
celery_app = Celery(__name__, broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL)


@celery_app.task(bind=True, base=Task)
def run_inference(self, program_string: str, inputs: dict, guidance_kwargs: dict, output_keys: List[str]) -> Dict[str, str]:
    print(
        f"Running inference for task ID: {self.request.id} on worker: {self.request.hostname}"
    )
    model_name = guidance_kwargs["llm"]["model_name"]
    del guidance_kwargs["llm"]
    program_string = clean_program_string(program_string)
    program = guidance(program_string, llm=get_llm(model_name), **guidance_kwargs)
    program_output = program(**inputs)
    outputs = {key: program_output[key] for key in output_keys}
    return outputs

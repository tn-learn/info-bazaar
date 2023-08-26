import os.path
from typing import List, Dict

import shutil
import guidance
import logging
from celery import Celery, Task

from bazaar.lem_utils import get_llm, clean_program_string
from llamapi import (
    REDIS_BROKER_URL,
    REDIS_BACKEND_URL,
    LLAMAPI_LOCAL_HF_CACHE_DIRECTORY,
    LLAMAPI_GLOBAL_HF_CACHE_DIRECTORY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery setup using the configurations from __init__.py
celery_app = Celery(__name__, broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL)


MODEL_DIRECTORIES = {
    "Llama-2-70b-chat-hf": "models--meta-llama--Llama-2-70b-chat-hf",
}


def ensure_model_available(model_name):
    """
    Ensure that the model specified by model_name is available.
    If not, it will copy or download the model.
    """
    # Get the path from MODEL_PATHS
    model_directory = MODEL_DIRECTORIES.get(model_name)

    if model_directory:
        # Define your cache directory or destination path
        src_path = os.path.join(LLAMAPI_GLOBAL_HF_CACHE_DIRECTORY, model_directory)
        dest_path = os.path.join(LLAMAPI_LOCAL_HF_CACHE_DIRECTORY, model_directory)
        # Determine if copy is needed
        copy_needed = src_path != dest_path
        if not copy_needed:
            logger.info(
                f"Copy not needed for {model_name} because the source path "
                f"{src_path} is the same as the destination path {dest_path}."
            )
        # Check if model already exists in the destination
        # If not, copy it
        if not os.path.exists(dest_path) and copy_needed:
            logger.info(f"Copying model {model_name} from {src_path} to {dest_path}")
            shutil.copytree(src_path, dest_path)
        else:
            logger.info(f"Found model {model_name} in {dest_path}")
    else:
        logger.error(f"Model {model_name} not found in MODEL_PATHS")
        raise ValueError(f"Model {model_name} not found in MODEL_PATHS")


@celery_app.task(bind=True, base=Task)
def run_inference(
    self,
    program_string: str,
    inputs: dict,
    guidance_kwargs: dict,
    output_keys: List[str],
) -> Dict[str, str]:
    logger.info(
        f"Running inference for task ID: {self.request.id} on worker: {self.request.hostname}"
    )
    try:
        # Make sure the model is all setup
        model_name = guidance_kwargs["llm"]["model_name"]
        guidance_kwargs.pop("llm")
        ensure_model_available(model_name)
        # Call guidance
        program_string = clean_program_string(program_string)
        program = guidance(  # noqa
            program_string, llm=get_llm(model_name), **guidance_kwargs
        )
        program_output = program(**inputs)
        outputs = {key: program_output[key] for key in output_keys}
        return outputs
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        raise e

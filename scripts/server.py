import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from bazaar.lem_utils import get_llm
import shutil
from typing import List, Union

def copy_model_if_not_exists(model_name):
    # Modify model_name by adding prefix and replacing / with --
    modified_model_name = "models--" + model_name.replace("/", "--")
    # Get the environment variables
    slurm_tmpdir = os.getenv("SLURM_TMPDIR")
    hf_cache_directory = os.getenv("HF_CACHE_DIRECTORY")

    # Construct the paths
    source_model_path = os.path.join(hf_cache_directory, modified_model_name)
    destination_model_path = os.path.join(slurm_tmpdir, "hf_home", modified_model_name)

    # Check if the model exists in the destination directory
    if not os.path.exists(destination_model_path):
        # If not, copy the model from the source directory
        shutil.copytree(source_model_path, destination_model_path)
        print(f"Copied {modified_model_name} to {destination_model_path}")
    else:
        print(f"{modified_model_name} already exists in {destination_model_path}")

    # Update the HF_CACHE_DIRECTORY environment variable
    os.environ["HF_CACHE_DIRECTORY"] = os.path.join(slurm_tmpdir, "hf_home")
    os.environ["HF_HOME"] = os.path.join(slurm_tmpdir, "hf_home")
    print(f"Updated HF_CACHE_DIRECTORY to {os.environ['HF_CACHE_DIRECTORY']}")



app = FastAPI()

class APIQuery(BaseModel):
    inputs: List[List[Union[int, float]]]
    temperature: float
    max_new_tokens: int
    top_p: float
    pad_token_id: int
    output_scores: bool
    return_dict_in_generate: bool

class APIResponse(BaseModel):
    response: str

@app.post("/generate", status_code=200)
def get_prediction(query: APIQuery):
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    copy_model_if_not_exists(model_name)
    llm = get_llm("Llama-2-70b-chat-hf")
    inputs_tensor = torch.tensor(query.inputs)

    # Call the generate method with the appropriate arguments
    result = llm.model_obj.generate(
        inputs=inputs_tensor,
        temperature=query.temperature,
        max_new_tokens=query.max_new_tokens,
        top_p=query.top_p,
        pad_token_id=query.pad_token_id,
        output_scores=query.output_scores,
        return_dict_in_generate=query.return_dict_in_generate
    )

    if query.return_dict_in_generate:
        scores = []
        if result.scores is not None:
            scores = result.scores.tolist()
        response = {"sequences": result.sequences.tolist(), "scores": scores}
    else:
        response = {"sequences": result.tolist()}
    return response

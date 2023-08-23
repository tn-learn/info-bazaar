import json
import os
from typing import Dict, List, Optional

import requests
import time

from llamapi import HOST_URL


def ask_for_guidance(
    program_string: str,
    *,
    inputs: Dict[str, str],
    output_keys: List[str],
    host_url: Optional[str] = None,
    **guidance_kwargs,
):
    if host_url is None:
        host_url = HOST_URL
    llm = guidance_kwargs.get("llm")
    if getattr(llm, "use_remote_guidance", False):
        url = f"{host_url}/guidance/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        guidance_kwargs["llm"] = llm.get_json_identifier()
        data = json.dumps(
            {
                "program_string": program_string,
                "inputs": inputs,
                "guidance_kwargs": guidance_kwargs,
                "output_keys": output_keys,
            }
        )
        response = requests.post(url, json={"payload": data}, headers=headers)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        response_data = response.json()

        def poll_for_results(task_id):
            while True:
                result_response = requests.get(f"{host_url}/results/{task_id}")
                result_data = result_response.json()
                status = result_data.get("status")

                if status == "Success":
                    return result_data.get("result")
                elif status == "Pending":
                    print(f"Task ID {task_id}: Still pending...")
                    time.sleep(2)  # Wait for 2 seconds before polling again
                else:
                    print("Failed to get results.")
                    break

        response_data = poll_for_results(response_data.get("task_id"))
        return response_data
    else:
        program = guidance(program_string, **guidance_kwargs)  # noqa
        program_output = program(**inputs)
        return {key: program_output[key] for key in output_keys}

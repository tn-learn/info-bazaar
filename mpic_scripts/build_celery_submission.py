# build_celery_submission.py

import os
import sys

# Get the current script's directory
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Get the worker number from command line arguments
worker_num = sys.argv[1]


# Submission file content template
template = f"""
get_env         = True
executable      = {DIR_PATH}/celery_worker_launcher.sh
output          = {os.environ['LLAMAPI_LOGS_DIR']}/celery_out_worker_{worker_num}.txt
error           = {os.environ['LLAMAPI_LOGS_DIR']}/celery_err_worker_{worker_num}.txt
log             = {os.environ['LLAMAPI_LOGS_DIR']}/celery_log_worker_{worker_num}.txt
request_cpus    = {os.environ['LLAMAPI_CELERY_CPUS']}
request_memory  = {os.environ['LLAMAPI_CELERY_RAM_MB']}
request_gpus    = {os.environ['LLAMAPI_CELERY_GPUS']}
requirements    = TARGET.CUDAGlobalMemoryMb > {os.environ['LLAMAPI_GPU_MIN_VRAM_MB']}
request_disk    = {os.environ['LLAMAPI_CELERY_DISK_GB']}G
queue
"""

with open(
    f"{os.environ['LLAMAPI_SUBMIT_FILES_DIR']}/celery_worker_{worker_num}.submit", "w"
) as f:
    f.write(template)

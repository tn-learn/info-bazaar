#!/bin/bash

# Configuration Environment Variables
export LLAMAPI_REDIS_BROKER_URL=${LLAMAPI_REDIS_BROKER_URL:-redis://localhost:6379/0}
export LLAMAPI_REDIS_BACKEND_URL=${LLAMAPI_REDIS_BACKEND_URL:-redis://localhost:6379/1}
export LLAMAPI_API_HOST=${LLAMAPI_API_HOST:-0.0.0.0}
export LLAMAPI_API_PORT=${LLAMAPI_API_PORT:-8000}
export LLAMAPI_NUM_CELERY_WORKERS=${LLAMAPI_NUM_CELERY_WORKERS:-2}

# GPU type parameter (default to rtx8000 if not specified)
GPU_TYPE=${1:-rtx8000}

# Temporary file to store cluster IDs (with a unique timestamp)
CLUSTER_ID_FILE="/tmp/llamapi_cluster_ids_$(date +%s).txt"
echo "" > $CLUSTER_ID_FILE

# Function to cleanup resources when script is terminated
cleanup() {
    echo "Cleaning up..."

    # Cancel the SLURM jobs
    while read -r JOB_ID
    do
        scancel $JOB_ID
    done < $CLUSTER_ID_FILE

    # Cleanup the temporary file
    rm -f $CLUSTER_ID_FILE

    exit
}

# Trap the SIGINT signal (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Start Redis server without protection
redis-server --protected-mode no &

# Give Redis some time to initialize
sleep 5

# Launch FastAPI server in the background
python -m llamapi.server &

# Give the server some time to initialize
sleep 5

# Submit jobs to SLURM to launch Celery workers and store the JOB_ID
for i in $(seq 1 $LLAMAPI_NUM_CELERY_WORKERS); do
    JOB_ID=$(salloc --gres=gpu:${GPU_TYPE}:1 --mem 32G -c 6 --partition long | grep -oP "Granted job allocation \K\d+")
    echo $JOB_ID >> $CLUSTER_ID_FILE
done

# Keep the script running to maintain the FastAPI and Redis servers
wait

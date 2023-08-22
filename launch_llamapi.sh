#!/bin/bash

# Configuration Environment Variables
export LLAMAPI_REDIS_BROKER_URL=${LLAMAPI_REDIS_BROKER_URL:-redis://localhost:6379/0}
export LLAMAPI_REDIS_BACKEND_URL=${LLAMAPI_REDIS_BACKEND_URL:-redis://localhost:6379/1}
export LLAMAPI_API_HOST=${LLAMAPI_API_HOST:-0.0.0.0}
export LLAMAPI_API_PORT=${LLAMAPI_API_PORT:-8000}
export LLAMAPI_NUM_CELERY_WORKERS=${LLAMAPI_NUM_CELERY_WORKERS:-2}

# Temporary file to store cluster IDs (with a unique timestamp)
CLUSTER_ID_FILE="/tmp/llamapi_cluster_ids_$(date +%s).txt"
echo "" > $CLUSTER_ID_FILE

# Function to cleanup resources when script is terminated
cleanup() {
    echo "Cleaning up..."

    # Kill the Condor jobs
    while read -r CLUSTER_ID
    do
        condor_rm $CLUSTER_ID
    done < $CLUSTER_ID_FILE

    # Cleanup the temporary file
    rm -f $CLUSTER_ID_FILE

    # You can add more cleanup tasks if needed
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

# Submit jobs to HTCondor to launch Celery workers and store the CLUSTER_ID
for i in $(seq 1 $LLAMAPI_NUM_CELERY_WORKERS); do
    condor_submit celery_worker.submit | grep -oP "submitted to cluster \K\d+" >> $CLUSTER_ID_FILE
done

# Keep the script running to maintain the FastAPI and Redis servers
wait

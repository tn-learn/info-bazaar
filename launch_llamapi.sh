#!/bin/bash

# Configuration Environment Variables
export LLAMAPI_REDIS_BROKER_URL=redis://localhost:6379/0
export LLAMAPI_REDIS_BACKEND_URL=redis://localhost:6379/1
export LLAMAPI_API_HOST=0.0.0.0
export LLAMAPI_API_PORT=8000

# Function to cleanup resources when script is terminated
cleanup() {
    echo "Cleaning up..."

    # Kill the Condor jobs (assuming you named the cluster "llamapi_worker_cluster")
    condor_rm llamapi_worker_cluster

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

# Submit jobs to HTCondor to launch Celery workers
# Note: Capture the Cluster ID for later usage in cleanup
CLUSTER_ID=$(condor_submit celery_worker.submit | grep -oP "submitted to cluster \K\d+")

# Store the cluster name for cleanup
echo "llamapi_worker_cluster" > ~/.condor_llamapi_cluster_name

# Keep the script running to maintain the FastAPI and Redis servers
wait

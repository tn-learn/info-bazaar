#!/bin/bash

echo "Setting up environment variables..."
# Configuration Environment Variables
# Redis
CURRENT_IP=$(hostname -I | awk '{print $1}')
export LLAMAPI_REDIS_BROKER_URL=${LLAMAPI_REDIS_BROKER_URL:-redis://$CURRENT_IP:6379/0}
export LLAMAPI_REDIS_BACKEND_URL=${LLAMAPI_REDIS_BACKEND_URL:-redis://$CURRENT_IP:6379/1}
export LLAMAPI_REDIS_SERVER_EXECUTABLE=${LLAMAPI_REDIS_SERVER_EXECUTABLE:-/home/nrahaman/utils/redis-stable/src/redis-server}
export LLAMAPI_REDIS_CLI_EXECUTABLE=${LLAMAPI_REDIS_CLI_EXECUTABLE:-/home/nrahaman/utils/redis-stable/src/redis-cli}
REDIS_PID=""

# FastAPI
export LLAMAPI_API_HOST=${LLAMAPI_API_HOST:-0.0.0.0}
export LLAMAPI_API_PORT=${LLAMAPI_API_PORT:-8000}
FASTAPI_PID=""

# Celery
export LLAMAPI_NUM_CELERY_WORKERS=${LLAMAPI_NUM_CELERY_WORKERS:-2}
export LLAMAPI_CELERY_CPUS=${LLAMAPI_CELERY_CPUS:-2}                # Default to 2 CPUs
export LLAMAPI_CELERY_RAM_MB=${LLAMAPI_CELERY_RAM_MB:-10000}        # Default to 100 GB
export LLAMAPI_CELERY_GPUS=${LLAMAPI_CELERY_GPUS:-1}                # Default to 1 GPUs
export LLAMAPI_CELERY_DISK_GB=${LLAMAPI_CELERY_DISK_GB:-160}        # Default to 160 GB
export LLAMAPI_GPU_MIN_VRAM_MB=${LLAMAPI_GPU_MIN_VRAM_MB:-70000}    # Default to a100
export LLAMAPI_CONDOR_BID=${LLAMAPI_CONDOR_BID:-2000}               # Default bid amount

# Huggingface
export LLAMAPI_GLOBAL_HF_CACHE_DIR=${LLAMAPI_GLOBAL_HF_CACHE_DIR:-/fast/nrahaman/persistmp/huggingface_cache}
export LLAMAPI_LOCAL_HF_CACHE_DIR=${LLAMAPI_LOCAL_HF_CACHE_DIR:-/tmp/huggingface_cache}

# Logistics
export LLAMAPI_TMPDIR=${LLAMAPI_TMPDIR:-/fast/nrahaman/persistmp/llamapi}
export LLAMAPI_SESSION_ID=$(date +%s)

# This is the tmp dir for this session only
SESSION_TMP_DIR="$LLAMAPI_TMPDIR/$LLAMAPI_SESSION_ID"

# Print the configuration
echo "==========================================="
echo "LlamAPI Configuration:"
echo "-------------------------------------------"
echo "Redis Broker URL: $LLAMAPI_REDIS_BROKER_URL"
echo "Redis Backend URL: $LLAMAPI_REDIS_BACKEND_URL"
echo "FastAPI Host: $LLAMAPI_API_HOST"
echo "FastAPI Port: $LLAMAPI_API_PORT"
echo "Number of Celery Workers: $LLAMAPI_NUM_CELERY_WORKERS"
echo "Celery CPUs: $LLAMAPI_CELERY_CPUS"
echo "Celery RAM (MB): $LLAMAPI_CELERY_RAM_MB"
echo "Celery GPUs: $LLAMAPI_CELERY_GPUS"
echo "Celery Disk (GB): $LLAMAPI_CELERY_DISK_GB"
echo "GPU Minimum VRAM (MB): $LLAMAPI_GPU_MIN_VRAM_MB"
echo "Condor Bid: $LLAMAPI_CONDOR_BID"
echo "Global Huggingface Cache Directory: $LLAMAPI_GLOBAL_HF_CACHE_DIR"
echo "Local Huggingface Cache Directory: $LLAMAPI_LOCAL_HF_CACHE_DIR"
echo "TMP Directory: $LLAMAPI_TMPDIR"
echo "Session ID: $LLAMAPI_SESSION_ID"
echo "Session TMP Directory: $SESSION_TMP_DIR"
echo "==========================================="


# Create the subs and logs directories
export LLAMAPI_SUBMIT_FILES_DIR="$SESSION_TMP_DIR/submit_files"
mkdir -p $LLAMAPI_SUBMIT_FILES_DIR
export LLAMAPI_LOGS_DIR="$SESSION_TMP_DIR/logs"
mkdir -p $LLAMAPI_LOGS_DIR

# Temporary file to store cluster IDs (with a unique timestamp)
CLUSTER_ID_FILE="$SESSION_TMP_DIR/llamapi_cluster_ids.txt"
touch $CLUSTER_ID_FILE

# Function to cleanup resources when script is terminated
cleanup() {
    echo "Cleaning up. This will take a short while."

    # Kill FastAPI
    kill $FASTAPI_PID

    # Wait a bit to ensure FastAPI has fully terminated
    sleep 5

    # Kill the Condor jobs (this will terminate the Celery workers)
    while read -r CLUSTER_ID
    do
        condor_rm $CLUSTER_ID
    done < $CLUSTER_ID_FILE

    # Wait for a short duration to allow the Celery workers to shut down gracefully
    sleep 15

    # Kill the Redis server if it was started by this script
    if [ ! -z "$REDIS_PID" ]; then
        kill $REDIS_PID
    fi

    # Cleanup the temporary file
    rm -f $CLUSTER_ID_FILE

    # You can add more cleanup tasks if needed
    exit
}


# Trap the SIGINT signal (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

echo "Starting Redis server..."
# Check if Redis is already running
if ! $LLAMAPI_REDIS_CLI_EXECUTABLE ping > /dev/null 2>&1; then
    $LLAMAPI_REDIS_SERVER_EXECUTABLE --protected-mode no &
    REDIS_PID=$!

    # Give Redis some time to initialize
    sleep 5
else
    echo "Redis is already running!"
fi

# Give Redis some time to initialize
sleep 5

# Launch FastAPI server in the background
python -m llamapi.server &

# Give the server some time to initialize
sleep 5

# Get the current script's directory
DIR_PATH=$(dirname $(realpath $0))

# Make sure that the celery launcher script is executable
chmod +x $DIR_PATH/celery_worker_launcher.sh

# Submit jobs to HTCondor to launch Celery workers and store the CLUSTER_ID
for i in $(seq 1 $LLAMAPI_NUM_CELERY_WORKERS); do
    echo "Generating submission file for worker $i..."
    # Generate a unique submission file for this worker
    python $DIR_PATH/build_celery_submission.py $i

    echo "Submitting worker $i to HTCondor..."
    # Submit the job to HTCondor using the generated submission file
    condor_submit_bid $LLAMAPI_CONDOR_BID $LLAMAPI_SUBMIT_FILES_DIR/celery_worker_$i.submit | grep -oP "submitted to cluster \K\d+" >> $CLUSTER_ID_FILE
done

# Print the address for the client to connect
echo "=============================================="
echo "LlamAPI is up and running!"
echo "Clients should connect to: http://$CURRENT_IP:$LLAMAPI_API_PORT"
echo "=============================================="

# Keep the script running to maintain the FastAPI and Redis servers
wait

#!/bin/bash

echo "Setting up environment variables..."
# Configuration Environment Variables
# Redis
export LLAMAPI_REDIS_BROKER_URL=${LLAMAPI_REDIS_BROKER_URL:-redis://0.0.0.0:6379/0}
export LLAMAPI_REDIS_BACKEND_URL=${LLAMAPI_REDIS_BACKEND_URL:-redis://0.0.0.0:6379/1}
export LLAMAPI_REDIS_SERVER_EXECUTABLE=${LLAMAPI_REDIS_SERVER_EXECUTABLE:-/home/nrahaman/utils/redis-stable/src/redis-server}
export LLAMAPI_REDIS_CLI_EXECUTABLE=${LLAMAPI_REDIS_CLI_EXECUTABLE:-/home/nrahaman/utils/redis-stable/src/redis-cli}

# FastAPI
export LLAMAPI_API_HOST=${LLAMAPI_API_HOST:-0.0.0.0}
export LLAMAPI_API_PORT=${LLAMAPI_API_PORT:-8000}

# Celery
export LLAMAPI_NUM_CELERY_WORKERS=${LLAMAPI_NUM_CELERY_WORKERS:-2}
export LLAMAPI_CELERY_CPUS=${LLAMAPI_CELERY_CPUS:-2}             # Default to 2 CPUs
export LLAMAPI_CELERY_RAM_MB=${LLAMAPI_CELERY_RAM_MB:-10000}     # Default to 100 GB
export LLAMAPI_CELERY_GPUS=${LLAMAPI_CELERY_GPUS:-1}             # Default to 1 GPUs
export LLAMAPI_CELERY_DISK_GB=${LLAMAPI_CELERY_DISK_GB:-160}     # Default to 160 GB
export LLAMAPI_GPU_MIN_VRAM_MB=${LLAMAPI_GPU_MIN_VRAM_MB:-70}    # Default to a100

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

echo "Starting Redis server..."
# Check if Redis is already running
if ! $LLAMAPI_REDIS_CLI_EXECUTABLE ping > /dev/null 2>&1; then
    $LLAMAPI_REDIS_SERVER_EXECUTABLE --protected-mode no &
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

# Submit jobs to HTCondor to launch Celery workers and store the CLUSTER_ID
for i in $(seq 1 $LLAMAPI_NUM_CELERY_WORKERS); do
    echo "Generating submission file for worker $i..."
    # Generate a unique submission file for this worker
    python $DIR_PATH/build_celery_submission.py $i

    echo "Submitting worker $i to HTCondor..."
    # Submit the job to HTCondor using the generated submission file
    condor_submit $LLAMAPI_SUBMIT_FILES_DIR/celery_worker_$i.submit | grep -oP "submitted to cluster \K\d+" >> $CLUSTER_ID_FILE
done

# Keep the script running to maintain the FastAPI and Redis servers
wait

#!/bin/bash

# Print the redis urls
echo "==========================================="
echo "Redis Broker URL: $LLAMAPI_REDIS_BROKER_URL"
echo "Redis Backend URL: $LLAMAPI_REDIS_BACKEND_URL"
echo "==========================================="

# Set the proxy
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=https://proxy:8080

# Create the cache directories in tmp
GUIDANCE_CACHE_DIR="/tmp/guidance_cache"
mkdir -p $GUIDANCE_CACHE_DIR

HUGGINGFACE_CACHE_DIR=$LLAMAPI_LOCAL_HF_CACHE_DIRECTORY
mkdir -p $HUGGINGFACE_CACHE_DIR

# Export the paths of these directories so that they can be accessed by other processes
export GUIDANCE_CACHE_DIRECTORY=$GUIDANCE_CACHE_DIR
export HF_CACHE_DIRECTORY=$HUGGINGFACE_CACHE_DIR
export HF_AUTH_TOKEN=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

# Get the directory where the script resides
SCRIPT_DIR=$(dirname $(realpath $0))

# Get the directory above the script's location and cd into it
# Call the script from the parent dir so llamapi is importable for reals
PARENT_DIR=$(dirname $SCRIPT_DIR)
cd $PARENT_DIR

echo "Starting celery worker..."
# Launch the celery worker (you can customize this command as needed)
celery -A llamapi.worker worker --loglevel=info

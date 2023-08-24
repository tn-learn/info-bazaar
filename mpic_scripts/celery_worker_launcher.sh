#!/bin/bash

# Define and create the guidance and huggingface cache directories in /tmp
GUIDANCE_CACHE_DIR="/tmp/guidance_cache"
HUGGINGFACE_CACHE_DIR="/tmp/huggingface_cache"
LLAMA2_PATH=/fast/nrahaman/persistmp/transformers_cache/models--meta-llama--Llama-2-70b-chat-hf

mkdir -p $GUIDANCE_CACHE_DIR
mkdir -p $HUGGINGFACE_CACHE_DIR

# Export the paths of these directories so that they can be accessed by other processes
export GUIDANCE_CACHE_DIRECTORY=$GUIDANCE_CACHE_DIR
export HF_CACHE_DIRECTORY=$HUGGINGFACE_CACHE_DIR
export HF_AUTH_TOKEN=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

echo "Copying LLaMa2 from $LLAMA2_PATH to $HUGGINGFACE_CACHE_DIR..."
# Copy the specified directory to the huggingface cache directory
cp -r $LLAMA2_PATH $HUGGINGFACE_CACHE_DIR/

echo "Starting celery worker..."
# Launch the celery worker (you can customize this command as needed)
celery -A llamapi.worker worker --loglevel=info

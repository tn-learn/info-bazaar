#!/bin/bash
#SBATCH --gres=gpu:$1:1
#SBATCH --mem 32G
#SBATCH -c 6
#SBATCH --partition long
#SBATCH --output=$SCRATCH/logs/celery_worker_%j.log

WORKER_IDX=$2

# llamapi exports
export LLAMAPI_LOCAL_HF_CACHE_DIRECTORY=$SLURM_TMPDIR/hf_home
mkdir -p $LLAMAPI_LOCAL_HF_CACHE_DIRECTORY

# Run the Celery worker with a unique name based on the index
module load anaconda/3
conda activate tn

export HF_CACHE_DIRECTORY=$LLAMAPI_LOCAL_HF_CACHE_DIRECTORY

export GUIDANCE_CACHE_DIRECTORY=$SLURM_TMPDIR/guidance_cache
mkdir -p $GUIDANCE_CACHE_DIRECTORY

# Get the directory where the script resides
SCRIPT_DIR=$(dirname $(realpath $0))

# Get the directory above the script's location and cd into it
# Call the script from the parent dir so llamapi is importable for reals
PARENT_DIR=$(dirname $SCRIPT_DIR)
cd $PARENT_DIR

celery -A llamapi.worker worker --loglevel=info --concurrency 1 -n worker${WORKER_IDX}@%h

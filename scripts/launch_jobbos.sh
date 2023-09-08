#!/bin/bash
#SBATCH --array=0-10
#SBATCH --job-name=my_job_array
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/mila/w/weissmar/scratch/tn/info-bazaar/logs/output_%A_%a.log
#SBATCH --error=/home/mila/w/weissmar/scratch/tn/info-bazaar/logs/error_%A_%a.log
# Experiment name passed as a script argument
EXPERIMENT_NAME=$1
CONFIG_FNAME=$2

# Generate unique output directory based on experiment name and array task ID
OUTPUT_DIR="/home/mila/w/weissmar/scratch/tn/info-bazaar/runs/${EXPERIMENT_NAME}/${SLURM_ARRAY_TASK_ID}"
CONFIG_PATH="/home/mila/w/weissmar/scratch/tn/info-bazaar/configs/${CONFIG_FNAME}"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

module load anaconda/3
source activate tn

export LLAMAPI_API_PORT=8911
export LLAMAPI_HOST_URL=http://login-1:8911

START=$((SLURM_ARRAY_TASK_ID))
END=$((START + 999))
STEP=10
echo  ${START}:${END}:${STEP}
python3 /home/mila/w/weissmar/scratch/tn/info-bazaar/scripts/main.py \
  --query_range ${START}:${END}:${STEP} \
  --embedding_manager_path /home/mila/w/weissmar/scratch/tn/info-bazaar/data/final_dataset_embeddings.db \
  --dataset_path /home/mila/w/weissmar/scratch/tn/info-bazaar/data/final_dataset_with_metadata.json \
  --output_path ${OUTPUT_DIR}
  --config ${CONFIG_PATH}
 
python3 /home/mila/w/weissmar/scratch/tn/info-bazaar/bazaar/eval.py \
  --run_directory ${OUTPUT_DIR}


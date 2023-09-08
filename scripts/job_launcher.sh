#!/bin/bash

EXPERIMENT_NAME=$1
CONFIG_FNAME=$2
GPU_TYPE=${3:-}

# Generate unique temporary script name
TEMP_SCRIPT="/tmp/sbatch_script_$(date +%s).sh"

# Write SBATCH directives to the temporary file
{
  echo "#!/bin/bash"
  echo "#SBATCH --array=0-2"
  echo "#SBATCH --job-name=my_job_array"
  echo "#SBATCH --cpus-per-task=4"
  echo "#SBATCH --mem=16G"
  if [ -n "$GPU_TYPE" ]; then
    echo "#SBATCH --gres=${GPU_TYPE}"
  fi
  echo "#SBATCH --output=\"/home/mila/w/weissmar/scratch/tn/info-bazaar/runs/${EXPERIMENT_NAME}/\${SLURM_ARRAY_TASK_ID}/logs/output_%A_%a.log\""
  echo "#SBATCH --error=\"/home/mila/w/weissmar/scratch/tn/info-bazaar/runs/${EXPERIMENT_NAME}/\${SLURM_ARRAY_TASK_ID}/logs/error_%A_%a.log\""

  echo "export EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  echo "export CONFIG_FNAME=${CONFIG_FNAME}"
} > ${TEMP_SCRIPT}

# Append your original script's content to the temporary file
cat >> ${TEMP_SCRIPT} << 'EOF'
echo "booting up."
module load anaconda/3
source activate tn

# Generate unique output directory based on experiment name and array task ID
OUTPUT_DIR="/home/mila/w/weissmar/scratch/tn/info-bazaar/runs/${EXPERIMENT_NAME}/${SLURM_ARRAY_TASK_ID}"
CONFIG_PATH="/home/mila/w/weissmar/scratch/tn/info-bazaar/configs/${CONFIG_FNAME}"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

module load anaconda/3
source activate tn

export LLAMAPI_API_PORT=8911
export LLAMAPI_HOST_URL=http://login-1:8911

START=$((SLURM_ARRAY_TASK_ID))
END=$((START + 99))
STEP=10

python3 /home/mila/w/weissmar/scratch/tn/info-bazaar/scripts/main.py \
  --query_range ${START}:${END}:${STEP} \
  --embedding_manager_path /home/mila/w/weissmar/scratch/tn/info-bazaar/data/final_dataset_embeddings.db \
  --dataset_path /home/mila/w/weissmar/scratch/tn/info-bazaar/data/final_dataset_with_metadata.json \
  --output_path ${OUTPUT_DIR}
  --config ${CONFIG_PATH}

python3 /home/mila/w/weissmar/scratch/tn/info-bazaar/bazaar/eval.py \
  --run_directory ${OUTPUT_DIR}
EOF

# Submit the temporary script to SLURM
sbatch ${TEMP_SCRIPT}

# Remove the temporary script
echo ${TEMP_SCRIPT}
#rm ${TEMP_SCRIPT}

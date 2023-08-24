#!/bin/bash
#SBATCH --gres=gpu:$1:1
#SBATCH --mem 32G
#SBATCH -c 6
#SBATCH --partition long
#SBATCH --output=$SCRATCH/logs/celery_worker_%j.log

WORKER_IDX=$2

# Run the Celery worker with a unique name based on the index
module load anaconda/3
conda activate tn
if [ ! -d "$SLURM_TMPDIR/hf_home/" ]; then
  echo "copying hf_home to $SLURM_TMPDIR/hf_home"
  cp -r "$SCRATCH/hf_home/" "$SLURM_TMPDIR/hf_home/"
fi

export HF_CACHE_DIRECTORY=$SLURM_TMPDIR/hf_home/
export HF_HOME=$SLURM_TMPDIR/hf_home/
cd llamapi
celery -A worker.celery_app worker --loglevel=info -n worker${WORKER_IDX}@%h


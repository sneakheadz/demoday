#!/bin/bash

#SBATCH --job-name=llama3_wikitext_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/llama3_wikitext_%j.out
#SBATCH --error=logs/llama3_wikitext_%j.err

# --- Environment Setup ---
module purge
module load cuda/12.1   # Match CUDA in the Docker image
module load apptainer # Or 'singularity' - ** STILL NEEDED TO RUN DOCKER IMAGE **

# --- Paths and Image ---
export PROJECT_DIR="/home/training"
# ** CHANGE: Point to the Docker image URI **
# Replace with your Docker Hub username/repo, or private registry URI
export CONTAINER_IMAGE="docker://hsiaochuansu/slrumoperator:1.0"
export OUTPUT_DIR="${PROJECT_DIR}/results/llama3-8b-wikitext-finetuned"
export HF_CACHE_DIR="${PROJECT_DIR}/hf_cache"
export WANDB_CACHE_DIR="${PROJECT_DIR}/wandb_cache"
export WANDB_API_KEY="YOUR_WANDB_API_KEY" # Optional
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN" # Optional

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${WANDB_CACHE_DIR}" "${PROJECT_DIR}/logs"

# --- Accelerate Configuration ---
# (Accelerate/Slurm config remains the same as before)
NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-$(echo $SLURM_JOB_GPUS | awk -F',' '{print NF}')}
NUM_PROCESSES=$((SLURM_NNODES * NUM_GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "-------------------- JOB CONFIGURATION --------------------"
echo "Project Directory: ${PROJECT_DIR}"
# ** CHANGE: Show Docker Image URI **
echo "Container Image: ${CONTAINER_IMAGE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "HF Cache Directory: ${HF_CACHE_DIR}"
echo "WandB Cache Directory: ${WANDB_CACHE_DIR}"
echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "Running on nodes: ${SLURM_JOB_NODELIST}"
echo "Number of Nodes: ${SLURM_NNODES}"
echo "GPUs per Node: ${NUM_GPUS_PER_NODE}"
echo "Total Processes (GPUs): ${NUM_PROCESSES}"
echo "Master Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "------------------------------------------------------------"


# --- Run Training Job ---
# ** STILL using 'apptainer exec' or 'singularity exec' **
# It pulls and runs the Docker image from the specified URI
srun --export=ALL apptainer exec --nv \
    --bind ${PROJECT_DIR}:/workspace \
    --bind ${OUTPUT_DIR}:/output \
    --bind ${HF_CACHE_DIR}:/opt/hf_cache \
    --bind ${WANDB_CACHE_DIR}:/opt/wandb_cache \
    ${CONTAINER_IMAGE} \
    accelerate launch \
        # (Accelerate launch arguments remain the same as before)
        --num_processes ${NUM_PROCESSES} \
        --num_machines ${SLURM_NNODES} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --machine_rank ${SLURM_NODEID} \
        --mixed_precision bf16 \
        /workspace/train_llama3_wikitext.py \
        # (Python script arguments remain the same as before)
        --model_name_or_path "meta-llama/Meta-Llama-3-8B" \
        --dataset_name "wikitext" \
        # ... rest of the arguments

echo "Slurm job finished successfully."

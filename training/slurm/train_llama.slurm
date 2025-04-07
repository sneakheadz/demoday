#!/bin/bash
#SBATCH --job-name=train_llama
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=main


# Set container image
CONTAINER_IMAGE="/home/newtraining/testimage.sqsh"

# Set output path
OUTPUT_PATH="/path/to/output"

# Pull dataset from Hugging Face
DATASET_NAME="salesforce/wikitext"
DATASET_PATH="/workspace/data"

# Run Pyxis container with Enroot
srun --container-name=train_llama \
     --container-mounts="${OUTPUT_PATH}:/output" \
     --container-workdir="/workspace" \
     enroot start \
     --mount="${OUTPUT_PATH}:/output" \
     ${CONTAINER_IMAGE} \
     python /workspace/train.py \
     --model_name_or_path "llama-3.1-8b" \
     --dataset_name "${DATASET_NAME}" \
     --dataset_path "${DATASET_PATH}" \
     --output_dir "/output" \
     --per_device_train_batch_size 8 \
     --gradient_accumulation_steps 4 \
     --learning_rate 5e-5

# Download dataset from Hugging Face
python -m datasets download ${DATASET_NAME} --path ${DATASET_PATH}

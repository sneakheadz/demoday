#!/bin/bash

#---------------------------------------------------------------------
# SLURM Directives (Derived from spec.slurmOptions)
#---------------------------------------------------------------------
# These directives mirror the settings in the 'spec.slurmOptions' section of the Nebius SlurmJob YAML.
# Ensure these match the resource requests defined in the YAML.

#SBATCH --job-name=llama3-train             # Corresponds to spec.slurmOptions.job-name
#SBATCH --output=llama3-train_%j.out        # Standard Slurm output log (%j = job ID)
#SBATCH --error=llama3-train_%j.err         # Standard Slurm error log (%j = job ID)
#SBATCH --nodes=1                           # Corresponds to spec.slurmOptions.nodes
#SBATCH --ntasks-per-node=1                 # Corresponds to spec.slurmOptions.ntasks-per-node
#SBATCH --gres=gpu:1                        # Corresponds to spec.slurmOptions.gres (adjust type/count based on Nebius/Slurm setup, e.g., gpu:a100:1)
#SBATCH --cpus-per-task=8                   # Corresponds to spec.slurmOptions.cpus-per-task
#SBATCH --mem=64G                           # Corresponds to spec.slurmOptions.mem (e.g., 64GB)
#SBATCH --time=01:00:00                     # Corresponds to spec.slurmOptions.time
#SBATCH --partition=gpu-queue             # Corresponds to spec.slurmOptions.partition (UNCOMMENT AND SET according to your cluster/Nebius Slurm setup if needed)

#---------------------------------------------------------------------
# Environment Setup & Variables (Derived from spec.container.env and Volumes)
#---------------------------------------------------------------------
echo "=================================================="
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES" # Usually set by Slurm via --gres
echo "=================================================="
echo "--- Setting Up Environment Variables ---"

# --- Hugging Face Token (Corresponds to spec.container.env[name=HF_TOKEN].valueFrom.secretKeyRef) ---
# IMPORTANT: The YAML uses a Kubernetes secret. For this script, you MUST provide the token securely.
# Recommended Method: Export it in your shell BEFORE running sbatch:
#   export HF_TOKEN="hf_your_actual_token"
#   sbatch this_script.sh
# Slurm typically inherits exported variables. Ensure HF_TOKEN is available.
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN environment variable is not set. Please export it before running sbatch."
  exit 1
fi
# Note: We export it here again just to be explicit, although inheritance should work.
export HF_TOKEN

# --- Training Configuration (Corresponds to spec.container.env) ---
# These variables match the 'env' section in the YAML spec.
export MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
export DATASET_ID="huggingface/CodeAlpaca-20k"
export NUM_TRAIN_EPOCHS="1"
export BATCH_SIZE="2"                     # Per-device batch size
export GRAD_ACCUMULATION_STEPS="8"        # Gradient accumulation steps
export LEARNING_RATE="2e-5"

# --- Volume Mount Paths (Corresponds to spec.container.volumeMounts and PVCs) ---
# The YAML defines container paths and relies on Kubernetes/Operator to mount volumes (PVCs).
# This script needs EXPLICIT host paths mapped to those container paths.
export CACHE_DIR="/cache"                 # Container-internal path (matches spec.container.env/volumeMounts)
export OUTPUT_DIR="/results"              # Container-internal path (matches spec.container.env/volumeMounts)

# !! CRITICAL !! REPLACE THESE HOST PATHS !!
# Define the ACTUAL paths on the cluster's shared filesystem where data should be read/written.
# These conceptually map to the persistent volumes (PVCs) the YAML would use.
# Ensure these directories exist and have correct permissions.
HOST_CACHE_PATH="/path/on/shared/filesystem/for/hf-cache"     # Replace with your actual host path for Hugging Face cache
HOST_RESULTS_PATH="/path/on/shared/filesystem/for/training-results" # Replace with your actual host path for output results

# --- Print Environment Configuration ---
echo "MODEL_ID: $MODEL_ID"
echo "DATASET_ID: $DATASET_ID"
echo "NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUMULATION_STEPS: $GRAD_ACCUMULATION_STEPS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "CACHE_DIR (Container): $CACHE_DIR"
echo "OUTPUT_DIR (Container): $OUTPUT_DIR"
echo "HOST_CACHE_PATH (Host): $HOST_CACHE_PATH"
echo "HOST_RESULTS_PATH (Host): $HOST_RESULTS_PATH"
echo "HF_TOKEN is set: ${HF_TOKEN:+yes}"
echo "------------------------------------------"

# --- Cluster Environment Modules ---
# Depending on your cluster setup, you might need to load modules for CUDA drivers
# and the container runtime (Apptainer/Singularity). The Nebius environment might handle
# this differently, potentially making these steps unnecessary if using the YAML CRD.
echo "--- Loading Environment Modules (Adjust for your cluster) ---"
module purge                         # Start with a clean environment
# Example: Load CUDA toolkit compatible with the container and requested GPU
module load cuda/11.8                # Or cuda/12.1, etc. CHECK YOUR CLUSTER REQUIREMENTS!
# Load the container runtime module available on your system
module load apptainer/1.1            # Or singularity/x.y, adjust name and version

echo "Modules loaded:"
module list 2>&1 # List loaded modules (redirect stderr to stdout for logging)
echo "------------------------------------------"


#---------------------------------------------------------------------
# Container Execution (Derived from spec.container)
#---------------------------------------------------------------------
# Define the container image (from spec.container.image)
CONTAINER_IMAGE="docker://cr.eu-north1.nebius.cloud/e00tswm87737468yd8/trainingcontainer:1.0"

# Define the container runtime command (use 'apptainer' or 'singularity')
CONTAINER_CMD="apptainer" # Or "singularity"

# Define the command to run inside the container (from spec.container.command)
# Assumes WORKDIR /app is set in the Dockerfile (matches spec.container.workingDir)
CONTAINER_ENTRYPOINT="python /app/train.py"

echo "--- Starting Container Execution ---"
echo "Container Image: ${CONTAINER_IMAGE}"
echo "Container Runtime: ${CONTAINER_CMD}"
echo "Executing Command: ${CONTAINER_ENTRYPOINT}"
echo "Binding Host Path ${HOST_CACHE_PATH} to Container Path ${CACHE_DIR}"
echo "Binding Host Path ${HOST_RESULTS_PATH} to Container Path ${OUTPUT_DIR}"
echo "------------------------------------------"

# Execute the command inside the container using the defined runtime
# - `--nv` enables NVIDIA GPU access inside the container.
# - `--bind` maps the specified HOST paths to the CONTAINER paths.
# - The environment variables exported above should be available inside the container.
${CONTAINER_CMD} exec \
    --nv \
    --bind ${HOST_CACHE_PATH}:${CACHE_DIR} \
    --bind ${HOST_RESULTS_PATH}:${OUTPUT_DIR} \
    ${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}

# Capture the exit code of the container command
EXIT_CODE=$?

#---------------------------------------------------------------------
# Job Completion
#---------------------------------------------------------------------
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Training job finished successfully at $(date)"
else
  echo "!!! Training job FAILED with exit code $EXIT_CODE at $(date) !!!"
fi
echo "=================================================="

exit $EXIT_CODE
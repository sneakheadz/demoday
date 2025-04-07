#!/bin/bash
#SBATCH --job-name=llama3_train_pyxis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # One main task (handled by srun)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=16            # CPUs for the main task
#SBATCH --mem=128G                    # Memory for the node (within 187G limit)
#SBATCH --time=24:00:00
#SBATCH --output=logs/llama3_wikitext_%j.out # Ensure logs directory exists
#SBATCH --error=logs/llama3_wikitext_%j.err  # Ensure logs directory exists
#SBATCH --partition=main                  # CORRECTED Partition name

# ================= Pyxis Container Configuration =========================
# Specify the container image for Slurm/Pyxis to use
#SBATCH --container-image="docker://hsiaochuansu/slrumoperator:2.0" # Verify this image and tag

# Specify necessary bind mounts for Slurm/Pyxis
# Define host paths first for clarity
export PROJECT_DIR="/home/training"
export OUTPUT_DIR="<span class="math-inline">\{PROJECT\_DIR\}/results/llama3\-8b\-wikitext\-finetuned"
export HF\_CACHE\_DIR\="</span>{PROJECT_DIR}/hf_cache"
export WANDB_CACHE_DIR="<span class="math-inline">\{PROJECT\_DIR\}/wandb\_cache"
\# Ensure host directories exist BEFORE the job step starts
\# NOTE\: This runs on the node where the batch script \*starts\* \(often login node\)
\# If compute nodes have different filesystems, ensure these paths are accessible there\.
mkdir \-p "</span>{OUTPUT_DIR}" "<span class="math-inline">\{HF\_CACHE\_DIR\}" "</span>{WANDB_CACHE_DIR}" "<span class="math-inline">\{PROJECT\_DIR\}/logs"
\# Construct the mount string for the \#SBATCH directive
export MY\_MOUNTS\="</span>{PROJECT_DIR}:/workspace,<span class="math-inline">\{OUTPUT\_DIR\}\:/output,</span>{HF_CACHE_DIR}:/opt/hf_cache,<span class="math-inline">\{WANDB\_CACHE\_DIR\}\:/opt/wandb\_cache"
\#SBATCH \-\-container\-mounts\=</span>{MY_MOUNTS}
# ========================================================================

# ================= Environment Module Setup (NEEDS CLUSTER-SPECIFIC FIX) =============
echo "Initializing environment modules..."
# IMPORTANT: Add the command needed to initialize modules on YOUR cluster below!
# Examples (replace with actual command from admins/docs):
# source /etc/profile.d/modules.sh
# source /etc/profile
# Or potentially leave blank if #!/bin/bash -l works (but it didn't seem to)
# ---> ADD CLUSTER-SPECIFIC MODULE INITIALIZATION HERE <---

module purge || echo "module purge failed (module command might still be unavailable)"
module load cuda/12.1 || echo "Failed to load cuda/12.1" # Match container's CUDA base if possible
# module load apptainer # Not needed if using Pyxis directives like --container-image

echo "Listing loaded modules (if module command works):"
module list
# ========================================================================

# Optional: Export API keys/tokens if needed inside the container
# Note: --export=ALL on srun might handle this, or use --container-env
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"
# export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

echo "========= Starting Slurm Job ========="
echo "Date: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "Running on node list: ${SLURM_JOB_NODELIST}"
echo "Project Directory: ${PROJECT_DIR}"
echo "Container Mounts: ${MY_MOUNTS}"
# Display container image info from Slurm's perspective
scontrol show job <span class="math-inline">SLURM\_JOB\_ID \| grep Container \|\| echo "Could not get container info via scontrol"
\# \-\-\- Accelerate Configuration \-\-\-
\# Calculate number of processes based on Slurm allocation
NUM\_GPUS\_PER\_NODE\=</span>{SLURM_GPUS_ON_NODE:-1} # Default to 1 if var not set
NUM_NODES=<span class="math-inline">\{SLURM\_NNODES\:\-1\}
NUM\_PROCESSES\=</span>((NUM_NODES * NUM_GPUS_PER_NODE))
NODE_RANK=<span class="math-inline">\{SLURM\_NODEID\:\-0\} \# Rank of the node within the allocation
\# Determine Master Address
\# WARNING\: This scontrol command will likely fail due to the slurm\.conf access issue\!
echo "Attempting to determine MASTER\_ADDR using scontrol\.\.\."
MASTER\_ADDR\=</span>(scontrol show hostnames <span class="math-inline">SLURM\_JOB\_NODELIST \| head \-n 1\)
if \[ \-z "</span>{MASTER_ADDR}" ]; then
    echo "WARNING: Failed to get MASTER_ADDR via scontrol. Using hostname as fallback."
    MASTER_ADDR=$(hostname)
fi
MASTER_PORT=29500 # Keep default port

echo "-------------------- Accelerate Config --------------------"
echo "Number of Nodes: ${NUM_NODES}"
echo "GPUs per Node: ${NUM_GPUS_PER_NODE}"
echo "Total Processes (Total GPUs): ${NUM_PROCESSES}"
echo "Node Rank: ${NODE_RANK}"
echo "Master Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "------------------------------------------------------------"

# ================= CRITICAL EXTERNAL ISSUE ==============================
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!! WARNING: This script requires the Slurm configuration issue !!!"
echo "!!! (srun/scontrol cannot find /etc/slurm/slurm.conf on       !!!"
echo "!!! worker nodes) to be FIXED BY THE CLUSTER ADMINISTRATORS   !!!"
echo "!!! before srun commands can succeed.                         !!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
# ========================================================================


# --- Run Training Job via srun (using Slurm/Pyxis container integration) ---
echo "Launching accelerate via srun (inside Pyxis container)..."

# srun executes the command directly inside the container defined by #SBATCH directives
srun \
    --export=ALL \ # Propagate environment variables from this script's env
    accelerate launch \
        --num_processes ${NUM_PROCESSES} \
        --num_machines ${NUM_NODES} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --machine_rank ${NODE_RANK} \
        --mixed_precision bf16 \
        # --use_fsdp \ # Uncomment if using FSDP
        /workspace/train_llama3_wikitext.py \
            --model_name_or_path "meta-llama/Meta-Llama-3-8B" \
            --dataset_name "wikitext" \
            --dataset_config_name "wikitext-103-raw-v1" \
            --output_dir "/output" \
            --do_train \
            --do_eval \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --per_device_eval_batch_size 8 \
            --learning_rate 2e-5 \
            --num_train_epochs 1 \
            --logging_steps 10 \
            --save_steps 100 \
            --evaluation_strategy "steps" \
            --eval_steps 100 \
            --save_total_limit 2 \
            --bf16 True \
            --tf32 True \
            --gradient_checkpointing True \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.03 \
            --weight_decay 0.01 \
            --block_size 1024 \
            --use_qlora True \
            --lora_r 16 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --preprocessing_num_workers 8 \
            --seed 42 \
            # --report_to "wandb" \
            # --wandb_project "llama3-8b-wikitext-finetune" \
            # --hf_token ${HF_TOKEN} \ # Requires HF_TOKEN to
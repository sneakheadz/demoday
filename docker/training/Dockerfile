# Use a base image with PyTorch, CUDA, and CUDNN matching requirements
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for git and potentially some pip installs
# Combine steps to reduce layers and clean up in the same layer
# REMOVED slurm-client
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages using pip
# Pin versions for reproducibility
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir \
    transformers==4.39.3 \
    datasets==2.18.0 \
    accelerate==0.29.3 \
    peft==0.10.0 \
    bitsandbytes==0.43.1 \
    torch==2.1.0 \
    wandb==0.16.6 \
    scipy

# Create directories for caching (optional, but good if mounting volumes later)
# Grant permissions widely, assuming non-root user might run container or mount volumes
RUN mkdir -p /opt/hf_cache /opt/wandb_cache && \
    chmod -R 777 /opt/hf_cache /opt/wandb_cache

# Set default environment variables for Hugging Face and WandB caches
# These can be overridden at runtime
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HOME=/opt/hf_cache
ENV WANDB_CACHE_DIR=/opt/wandb_cache
ENV WANDB_DISABLED=true

# REMOVED slurm.conf copying and chmod

# Copy python script (optional if relying on bind mount, but ok to keep)
COPY train_llama3_wikitext.py /workspace/train_llama3_wikitext.py

# Set a working directory for the container
WORKDIR /workspace

# Default command (can be easily overridden when running the container)
# Useful for basic checks or if the user always runs python scripts
CMD ["python"]
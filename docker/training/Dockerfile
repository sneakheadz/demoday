# Use a base image with CUDA and PyTorch matching your needs
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install Python dependencies
# Use recent versions of transformers/datasets/accelerate
RUN pip install --no-cache-dir \
    transformers[torch]>=4.35.0 \
    datasets>=2.15.0 \
    accelerate>=0.25.0 \
    tensorboard
    # Add other needed libs below here if necessary, ensuring the last one
    # does NOT end with a backslash (\) unless followed by more packages.
    # ---
    # Example for potentially installing NVIDIA Apex (ensure source is present):
    # pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# Copy the training script into the image
# Ensure train.py is in the same directory as the Dockerfile during build
COPY training.py .

# Make port 8080 available for readiness probe if needed by Soperator
# EXPOSE 8080

# Entrypoint can be overridden in Soperator manifest
# ENTRYPOINT ["python", "train.py"]

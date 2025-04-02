# Use Ubuntu as the base image
FROM ubuntu:latest

# Update package lists and install dependencies (including Python and pip)
RUN apt-get update && apt-get install -y \
    wget unzip curl jq nano coreutils python3 python3-pip ca-certificates curl gnupg lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Install Terraform 1.11.3
RUN wget https://releases.hashicorp.com/terraform/1.11.3/terraform_1.11.3_linux_amd64.zip && \
    unzip terraform_1.11.3_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform_1.11.3_linux_amd64.zip

# Install Nebius CLI
RUN curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash

# Install kubectl
RUN curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/

# Install aws cli
RUN pip3 install --break-system-packages awscli

# Add Docker's official GPG key:
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository:
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package lists and install Docker CE
RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

#remove this later
RUN usermod -aG docker root


# Create installation folder
RUN mkdir -p /installations

# Copy installation file into the container
COPY installations/ /installations

# Verify installations (optional)
RUN terraform --version && \
    kubectl version --client && \
    jq --version && \
    which jq && \
    which terraform && \
    which kubectl && \
    aws --version && \
    docker --version

# Set the working directory (optional)
WORKDIR /installations

# Define the entry point (optional)
ENTRYPOINT ["/bin/bash"]
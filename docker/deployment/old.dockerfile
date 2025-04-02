# Use Ubuntu as the base image
FROM ubuntu:latest

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    jq \
    nano\
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Install Terraform 1.11.3
RUN wget https://releases.hashicorp.com/terraform/1.11.3/terraform_1.11.3_linux_amd64.zip \
    && unzip terraform_1.11.3_linux_amd64.zip \
    && mv terraform /usr/local/bin/ \
    && rm terraform_1.11.3_linux_amd64.zip

# Install Nebius CLI
RUN curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash 

# Install kubectl
RUN curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Update the package lists and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

# Add Docker's official GPG key
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# Add Docker's stable repository
RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Update package lists again and install Docker CE
RUN apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io

# Optionally, add the current user to the docker group
# This is usually not recommended for production images, but can be useful for development
RUN usermod -aG docker $USER

# Create insallation folder
RUN mkdir -p /installations

# Copy installation file into the container
COPY installations/ /installations

# Verify installations (optional)
RUN terraform --version \
   # && nebius version \
    && kubectl version --client \
    && jq --version \
    && which jq \
    && which terraform \
    && which kubectl 
   # && which nebius

# Set the working directory (optional)
WORKDIR /installations

# Define the entry point (optional)
ENTRYPOINT ["/bin/bash"]
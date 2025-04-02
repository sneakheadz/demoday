FROM ubuntu:latest

# Install base dependencies and necessary tools in a single RUN command
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget unzip curl jq nano coreutils apt-transport-https ca-certificates gnupg lsb-release \
    software-properties-common docker-ce docker-ce-cli containerd.io python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
    apt-get update && \
    curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/ && \
    wget https://releases.hashicorp.com/terraform/1.11.3/terraform_1.11.3_linux_amd64.zip && \
    unzip terraform_1.11.3_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform_1.11.3_linux_amd64.zip && \
    curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash && \
    mkdir /installations && \
    apt-get clean && \
    pip3 install awscli

COPY installations/ /installations/

# Verify installations (optional)
RUN terraform --version && \
    kubectl version --client && \
    jq --version && \
    which jq && \
    which terraform && \
    which kubectl && \
    aws --version

# Set the working directory
WORKDIR /installations

# Set entry point
ENTRYPOINT ["/bin/bash"]
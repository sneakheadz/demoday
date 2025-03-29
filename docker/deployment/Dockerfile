# Use Ubuntu 22.04 LTS as the base image
FROM ubuntu:22.04

# Set frontend to noninteractive to avoid prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Define arguments for tool versions (use 'latest' for kubectl, provide specific version for Terraform)
ARG KUBECTL_VERSION="latest"
ARG TERRAFORM_VERSION="1.8.0" # Specify a recent Terraform version
ARG TARGETARCH=amd64 # Default architecture, Docker buildx can override this

# Update package list and install dependencies
# curl, wget: for downloading binaries/scripts
# unzip: for Terraform archive
# gpg: for verifying Terraform signature
# ca-certificates: for HTTPS connections
# jq: JSON processor
# coreutils: Standard GNU core utilities (often already present, but ensures it)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    unzip \
    gpg \
    ca-certificates \
    jq \
    coreutils && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Install Terraform
RUN echo "Installing Terraform version: ${TERRAFORM_VERSION}" && \
    wget "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_${TARGETARCH}.zip" && \
    wget "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_SHA256SUMS" && \
    wget "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_SHA256SUMS.sig" && \
    # Import HashiCorp GPG key (check https://www.hashicorp.com/security#gpg-key for current key)
    gpg --keyserver keyserver.ubuntu.com --recv-keys 798A218F5A0F365F && \
    # Verify the signature
    gpg --verify "terraform_${TERRAFORM_VERSION}_SHA256SUMS.sig" "terraform_${TERRAFORM_VERSION}_SHA256SUMS" && \
    # Verify the checksum
    grep "terraform_${TERRAFORM_VERSION}_linux_${TARGETARCH}.zip" "terraform_${TERRAFORM_VERSION}_SHA256SUMS" | sha256sum -c - && \
    # Unzip and install
    unzip "terraform_${TERRAFORM_VERSION}_linux_${TARGETARCH}.zip" -d /usr/local/bin && \
    # Clean up downloaded files
    rm -f "terraform_${TERRAFORM_VERSION}_linux_${TARGETARCH}.zip" \
          "terraform_${TERRAFORM_VERSION}_SHA256SUMS" \
          "terraform_${TERRAFORM_VERSION}_SHA256SUMS.sig" && \
    # Verify installation
    terraform --version

# Install kubectl
RUN echo "Installing kubectl version: ${KUBECTL_VERSION}" && \
    if [ "${KUBECTL_VERSION}" = "latest" ]; then \
      KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt); \
    fi && \
    echo "Resolved kubectl version: ${KUBECTL_VERSION}" && \
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" && \
    # (Optional: Verify checksum - get checksum from https://dl.k8s.io/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl.sha256)
    # curl -LO "https://dl.k8s.io/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl.sha256" && \
    # echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm -f kubectl kubectl.sha256 && \
    # Verify installation
    kubectl version --client

# Install Nebius CLI
# Using the recommended installation method from Nebius docs (assumed to be similar to gcloud/aws)
RUN echo "Installing Nebius CLI..." && \
    curl -sSL https://storage.nemax.nebius.cloud/cli/install.sh | bash && \
    # The script likely installs to /root/nebius-cloud/bin, add it to PATH
    # Adjust the path if the install script places it elsewhere
    echo 'export PATH="/root/nebius-cloud/bin:$PATH"' >> /root/.bashrc && \
    # Verify installation by accessing the binary directly via expected path
    /root/nebius-cloud/bin/nebius --version

# Add Nebius CLI bin directory to the system PATH environment variable
# This ensures 'nebius' command is available directly
ENV PATH="/root/nebius-cloud/bin:${PATH}"

# Set a working directory (optional)
WORKDIR /work

# Default command when running the container (optional)
# CMD ["bash"]

# --- End of Dockerfile ---

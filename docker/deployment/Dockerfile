# Use Ubuntu as the base image
FROM ubuntu:latest

# Update package lists and install dependencies
# Added openssh-client
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    curl \
    jq \
    nano \
    coreutils \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Terraform 1.11.3
# Note: Using a specific version. Check for newer versions if needed.
RUN wget https://releases.hashicorp.com/terraform/1.11.3/terraform_1.11.3_linux_amd64.zip \
    && unzip terraform_1.11.3_linux_amd64.zip \
    && mv terraform /usr/local/bin/ \
    && rm terraform_1.11.3_linux_amd64.zip

# Install Nebius CLI
# Assuming the install script handles adding nebius to PATH, otherwise adjust verification
# The symlink helps ensure it's found in a standard PATH location.
RUN curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash \
    && ln -s /root/bin/nebius /usr/local/bin/nebius

# Install kubectl
# Fetches the latest stable version
RUN curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    # Run install script (installs to /usr/local/aws-cli, symlinks to /usr/local/bin)
    && ./aws/install \
    # Clean up installer files
    && rm -rf awscliv2.zip aws

# Create installation folder
RUN mkdir -p /installations

# Copy installation file into the container
# Ensure an 'installations' directory exists in your build context (next to the Dockerfile)
COPY installations/ /installations/

# Verify installations (optional but recommended)
# Chain commands with && so the build fails if any verification fails
RUN terraform --version \
    && kubectl version --client \
    && aws --version \
    && ssh -V \
    && jq --version \
    && which jq \
    && which terraform \
    && which kubectl \
    && which aws \
    && which ssh
    # Verify Nebius CLI - use the symlink path which should be in $PATH
    # Uncomment the line below if you want the build to fail if nebius isn't found/working
    # && nebius version

# Set the working directory (optional)
WORKDIR /installations/

# Define the entry point (optional)
# This makes the container run bash interactively when started without a command
ENTRYPOINT ["/bin/bash"]

# Note: This Dockerfile includes the SSH *client* (openssh-client).
# If you need an SSH *server* inside the container (to SSH *into* it),
# you would need to install 'openssh-
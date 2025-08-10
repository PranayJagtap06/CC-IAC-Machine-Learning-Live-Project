# syntax=docker/dockerfile:1
# Use the official Python 3.13 image based on Debian Bookworm
FROM python:3.13-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GEMINI_API_KEY='<enter your Google Gemini API Key>' \
    NVM_DIR="/usr/local/nvm" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=all

# Update apt package lists, install sudo, and essential build tools/libraries
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    sudo \
    build-essential iproute2 software-properties-common \
    libgl1-mesa-glx net-tools wl-clipboard \
    git curl wget fish tree unzip gnupg \
    axel aria2 htop btop luarocks \
    openssh-server ca-certificates \
    # Clean up APT cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the NVIDIA CUDA repo
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update

# Install CUDA Toolkit and cuDNN
RUN apt-get install -y --no-install-recommends \
    cuda-toolkit-12-3 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Add CUDA to PATH
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install latest stable Neovim (v0.10.0) required for modern plugins like NvChad
RUN curl -L -o /tmp/nvim.tar.gz https://github.com/neovim/neovim/releases/download/v0.11.3/nvim-linux-x86_64.tar.gz \
    && tar -xzf /tmp/nvim.tar.gz -C /usr/local/ \
    && ln -s /usr/local/nvim-linux64/bin/nvim /usr/local/bin/nvim \
    && rm /tmp/nvim.tar.gz

# Define build arguments for user creation (can be customized during build)
ARG USERNAME=ml-brain-dev
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Create a non-root user with a home directory and bash shell
# Add the user to the 'sudo' group for administrative privileges
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    # Grant passwordless sudo privileges to the new user
    && echo ${USERNAME} ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    # Set appropriate permissions for the sudoers file
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # Create SSH directory structure
    && mkdir -p /var/run/sshd \
    && mkdir -p /home/${USERNAME}/.ssh \
    && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.ssh \
    && chmod 700 /home/${USERNAME}/.ssh \
    # Create config directories with proper permissions
    && mkdir -p /home/${USERNAME}/.config \
    && mkdir -p /home/${USERNAME}/.config/matplotlib \
    && mkdir -p /home/${USERNAME}/.local/share \
    && mkdir -p /home/${USERNAME}/.cache \
    && mkdir -p /home/${USERNAME}/.jupyter \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.config \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.local \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.jupyter \
    && chmod -R 755 /home/${USERNAME}/.config \
    && chmod -R 755 /home/${USERNAME}/.local \
    && chmod -R 755 /home/${USERNAME}/.cache \
    && chmod -R 755 /home/${USERNAME}/.jupyter \
    # Backup original config (optional, but good practice)
    && mv /etc/ssh/sshd_config /etc/ssh/sshd_config.bak \
    # Create a minimal sshd_config for key-based auth
    && echo "Port 22" > /etc/ssh/sshd_config \
    && echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config \
    && echo "PermitRootLogin no" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config \
    && echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "AuthorizedKeysFile      .ssh/" >> /etc/ssh/sshd_config \
    && echo "Subsystem       sftp    /usr/lib/openssh/sftp-server" >> /etc/ssh/sshd_config \
    # Ensure correct permissions for sshd_config
    && chmod 644 /etc/ssh/sshd_config \
    \
    # Generate SSH Host Keys
    # This step ensures that sshd has the necessary keys to start up.
    # The -A option generates all default host key types.
    && ssh-keygen -A \
    && chmod 600 /etc/ssh/ssh_host_*_key \
    && chmod 644 /etc/ssh/ssh_host_*_key.pub

# SHELL ["/bin/bash", "-c"]

# Install Node.js LTS and update npm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest

# Set the working directory inside the container's user's home folder
WORKDIR /home/${USERNAME}

RUN curl -sS https://starship.rs/install.sh -o /tmp/starship-install.sh \
    && sh /tmp/starship-install.sh -y && rm /tmp/starship-install.sh \
    && echo "eval '$(starship init bash)'" >> /home/${USERNAME}/.bashrc \
    && mkdir -p /home/${USERNAME}/.config/fish && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.config/fish \
    && echo 'starship init fish | source' >> /home/${USERNAME}/.config/fish/config.fish \
    && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.config/fish/config.fish \
    && chsh -s /usr/bin/fish ${USERNAME} \
    && npm install -g @google/gemini-cli 
# && git clone https://github.com/PranayJagtap06/pj-nvchad-config.git /home/${USERNAME}/.config/nvim && nvim

# Switch to the newly created non-root user
USER ${USERNAME}

ENV HOME=/home/${USERNAME} \
    PATH=$HOME/.local/bin:$PATH \
    MPLCONFIGDIR=/home/${USERNAME}/.config/matplotlib \
    XDG_CONFIG_HOME=/home/${USERNAME}/.config \
    XDG_CACHE_HOME=/home/${USERNAME}/.cache \
    XDG_DATA_HOME=/home/${USERNAME}/.local/share \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=all \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

# Copy the requirements.txt file into the working directory
COPY --chown=${USERNAME}:${USERNAME} ./requirements.txt start.sh ./
COPY --chown=${USERNAME}:${USERNAME} ./id_ed25519.pub /home/${USERNAME}/.ssh/

# Ensure correct permissions for the file
RUN if [ -f /home/${USERNAME}/.ssh/id_ed25519.pub ]; then chmod 644 /home/${USERNAME}/.ssh/id_ed25519.pub; fi
# RUN chmod 600 /home/${USERNAME}/.ssh/*
RUN chown ${USERNAME}:${USERNAME} start.sh && chmod 755 start.sh

# Install Python dependencies with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    python3 -m venv ml-brain-env && \
    /bin/bash -c "source ml-brain-env/bin/activate && \
    pip install --no-cache-dir uv && \
    export UV_HTTP_TIMEOUT=300 && \
    uv pip install --no-cache-dir -r requirements.txt && \
    # Ensure PyTorch with CUDA support is installed
    uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126" && \
    # uv pip install torchmetrics 'tensorflow[and-cuda]'" && \
    echo "source /home/ml-brain-dev/ml-brain-env/bin/activate.fish" >> $HOME/.config/fish/config.fish && \
    echo "set -gx PYTHONPATH /home/ml-brain-dev:/home/ml-brain-dev/workspace" >> $HOME/.config/fish/config.fish && \
    echo "set -gx CUDA_VISIBLE_DEVICES all" >> $HOME/.config/fish/config.fish && \
    echo "set -gx NVIDIA_VISIBLE_DEVICES all" >> $HOME/.config/fish/config.fish && \
    mkdir -p $HOME/workspace

# Expose port 8888 for JupyterLab (optional, but common for AI/ML development)
EXPOSE 22 8888-8896

# Default command to run when the container starts.
CMD ["./start.sh"]


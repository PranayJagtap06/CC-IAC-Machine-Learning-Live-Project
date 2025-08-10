#!/bin/bash

echo "ML Brain Development Environment Ready!"
echo "Python virtual environment: ml-brain-env"
echo "Working directory: /home/ml-brain-dev"
echo "Available ports: 8888-8896"
echo "To activate the environment: source ml-brain-env/bin/activate"
echo "To start JupyterLab: source /home/ml-brain-dev/ml-brain-env/bin/activate && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""

echo "Updating & Upgrading container environment..."
sudo apt-get update && sudo apt-get upgrade -y

# Fix permissions for user directories
echo ""
echo "Fixing permissions..."
sudo chown -R ml-brain-dev:ml-brain-dev /home/ml-brain-dev
sudo chmod -R 755 /home/ml-brain-dev/.config
sudo chmod -R 755 /home/ml-brain-dev/.local
sudo chmod -R 755 /home/ml-brain-dev/.cache
sudo chmod 700 /home/ml-brain-dev/.ssh
sudo chmod 600 /home/ml-brain-dev/.ssh/authorized_keys

# Ensure matplotlib config directory exists and is writable
mkdir -p /home/ml-brain-dev/.config/matplotlib
chmod 755 /home/ml-brain-dev/.config/matplotlib

# Start the SSH daemon in the background
# This command initializes and runs the SSH server.
# /usr/sbin/sshd &

echo "Setting up SSH daemon..."

sudo bash -c '
    # Ensure SSH directory exists
    mkdir -p /var/run/sshd

    # Check if host keys exist, if not generate them
    if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
        echo "Generating SSH host keys..."
        ssh-keygen -A
        chmod 600 /etc/ssh/ssh_host_*_key
        chmod 644 /etc/ssh/ssh_host_*_key.pub
    fi

    # Verify SSH configuration
    echo "Testing SSH configuration..."
    /usr/sbin/sshd -t
    if [ $? -eq 0 ]; then
        echo "SSH configuration is valid"
    else
        echo "SSH configuration has errors!"
        exit 1
    fi

    # Start SSH daemon with debugging (remove -d for production)
    echo "Starting SSH daemon..."
    /usr/sbin/sshd -D -d &
    SSH_PID=$!
    echo "SSH daemon started with PID: $SSH_PID"

    # Wait a moment and check if SSH is running
    sleep 2
    if ps -p $SSH_PID > /dev/null; then
        echo "SSH daemon is running successfully"
        netstat -tlnp | grep :22 || ss -tlnp | grep :22
    else
        echo "Failed to start SSH daemon"
        exit 1
    fi
'

# Verify authorized_keys file
echo " "
echo "Checking SSH key setup..."
if [ -f /home/ml-brain-dev/.ssh/id_rsa.pub ]; then
    echo "Authorized keys file exists"
    ls -la /home/ml-brain-dev/.ssh/id_rsa.pub
    echo "Key fingerprints:"
    ssh-keygen -lf /home/ml-brain-dev/.ssh/id_rsa.pub
else
    echo "WARNING: No authorized_keys file found!"
fi

echo "SSH setup completed"
echo ""
echo "Connection details for Spyder:"
echo "  Host: localhost"
echo "  Port: 2222"
echo "  Username: ml-brain-dev"
echo "  SSH Key: Use the private key corresponding to id_rsa.pub"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
else
    echo "No NVIDIA GPU detected or nvidia-smi not available"
fi
echo ""

# Show environment variables for debugging
echo "Environment variables:"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"
echo "XDG_CONFIG_HOME: $XDG_CONFIG_HOME"
echo "HOME: $HOME"
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate virtual environment and check GPU in Python
echo "Checking GPU availability in Python..."
source /home/ml-brain-dev/ml-brain-env/bin/activate
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
"
echo ""

echo "Jupyter runtime directory:"
jupyter --runtime-dir
mkdir -p $(jupyter --runtime-dir)
ls -la $(jupyter --runtime-dir) 2>/dev/null || echo "Runtime directory not yet created"
echo ""

# Create Jupyter configuration directory and config file
echo "Setting up Jupyter configuration..."
mkdir -p /home/ml-brain-dev/.jupyter
cat > /home/ml-brain-dev/.jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab configuration file
c = get_config()

# Network configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True

# Security configuration - DISABLE PASSWORD/TOKEN FOR DEVELOPMENT
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# Allow all origins (for development only)
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_credentials = True

# Disable authentication entirely (ONLY for development environments)
c.IdentityProvider.token = ''

# Additional settings for better development experience
c.ServerApp.terminado_settings = {'shell_command': ['/usr/bin/fish']}
c.ServerApp.notebook_dir = '/home/ml-brain-dev/workspace'
c.ServerApp.root_dir = '/home/ml-brain-dev'

# Enable GPU monitoring extensions if available
c.ServerApp.jpserver_extensions = {
    'jupyter_server_proxy': True,
}

# Set environment variables for CUDA in notebooks - ALL GPUs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'all'
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
EOF

chown ml-brain-dev:ml-brain-dev /home/ml-brain-dev/.jupyter/jupyter_lab_config.py
chmod 644 /home/ml-brain-dev/.jupyter/jupyter_lab_config.py

echo "Jupyter configuration created with disabled authentication"
echo ""

# Start JupyterLab with the configuration
echo "Starting Jupyter Lab with GPU support and disabled authentication..."
source /home/ml-brain-dev/ml-brain-env/bin/activate && \
CUDA_VISIBLE_DEVICES=all \
NVIDIA_VISIBLE_DEVICES=all \
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True

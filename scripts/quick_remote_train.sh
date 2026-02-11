#!/bin/bash
# Quick Remote Training Script
# Usage: ./quick_remote_train.sh username gpu-server.university.edu

USERNAME=$1
SERVER=$2

if [ -z "$USERNAME" ] || [ -z "$SERVER" ]; then
    echo "Usage: $0 <username> <server>"
    echo "Example: $0 jdoe gpu.university.edu"
    exit 1
fi

echo "🚀 Starting remote training setup..."
echo "Server: $USERNAME@$SERVER"
echo ""

# 1. Transfer code
echo "📦 Transferring code..."
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '.DS_Store' \
  . $USERNAME@$SERVER:~/acie/

# 2. Transfer small dataset sample for testing
echo "📊 Transferring sample data..."
rsync -avz --progress lib/acie_observational_10k_x_10k.csv \
  $USERNAME@$SERVER:~/acie/lib/

# 3. Setup and start training
echo "🔧 Setting up environment and starting training..."
ssh $USERNAME@$SERVER << 'ENDSSH'
cd ~/acie
mkdir -p logs outputs/checkpoints

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Setup Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch pandas pyyaml tqdm numpy
else
    source venv/bin/activate
fi

# Verify GPU
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Start training in screen
screen -dmS acie_training bash -c "cd ~/acie && source venv/bin/activate && python3 scripts/train_simple.py 2>&1 | tee logs/training.log"

echo ""
echo "✅ Training started in screen session 'acie_training'"
echo ""
echo "To monitor:"
echo "  ssh $USERNAME@$SERVER"
echo "  screen -r acie_training  # Attach to session"
echo "  tail -f ~/acie/logs/training.log  # View logs"
echo ""
ENDSSH

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 To monitor training:"
echo "  ssh $USERNAME@$SERVER 'tail -f ~/acie/logs/training.log'"
echo ""
echo "🔍 To check GPU usage:"
echo "  ssh $USERNAME@$SERVER 'nvidia-smi'"
echo ""
echo "📥 To download trained model later:"
echo "  scp $USERNAME@$SERVER:~/acie/outputs/checkpoints/acie_best.pt outputs/checkpoints/"

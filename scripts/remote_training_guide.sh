#!/bin/bash
# ACIE Remote GPU Training Setup Guide

## Prerequisites
# 1. University GPU server credentials (username, hostname, password/key)
# 2. SSH access to the cluster

# ============================================================================
# STEP 1: TRANSFER CODE TO REMOTE SERVER
# ============================================================================

# From your Mac, sync ACIE project to remote server
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '.git' \
  /Users/jitterx/Desktop/ACIE/ \
  username@gpu-server.university.edu:~/acie/

# Transfer datasets (14GB - may take time)
rsync -avz --progress \
  /Users/jitterx/Desktop/ACIE/lib/*.csv \
  username@gpu-server.university.edu:~/acie/lib/

# ============================================================================
# STEP 2: SSH INTO GPU SERVER
# ============================================================================

ssh username@gpu-server.university.edu

# ============================================================================
# STEP 3: SETUP ENVIRONMENT ON REMOTE SERVER
# ============================================================================

# Navigate to project
cd ~/acie

# Check GPU availability
nvidia-smi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas pyyaml tqdm numpy scipy matplotlib

# Verify GPU is detected
python3 -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count())"

# ============================================================================
# STEP 4: START TRAINING (Background Process)
# ============================================================================

# Option A: Using nohup (simple)
nohup python3 scripts/train_simple.py > logs/training.log 2>&1 &
echo $! > training.pid

# Option B: Using screen (recommended - can reattach)
screen -S acie_training
python3 scripts/train_simple.py
# Press Ctrl+A then D to detach

# Option C: Using tmux (modern alternative)
tmux new -s acie_training
python3 scripts/train_simple.py
# Press Ctrl+B then D to detach

# ============================================================================
# STEP 5: MONITOR TRAINING (From Your Mac)
# ============================================================================

# Watch training log in real-time
ssh username@gpu-server.university.edu "tail -f ~/acie/outputs/training.log"

# Check GPU usage
ssh username@gpu-server.university.edu "watch -n 1 nvidia-smi"

# Reattach to screen session
ssh username@gpu-server.university.edu
screen -r acie_training

# ============================================================================
# STEP 6: RETRIEVE TRAINED MODEL
# ============================================================================

# After training completes, download the best model
scp username@gpu-server.university.edu:~/acie/outputs/checkpoints/acie_best.pt \
  /Users/jitterx/Desktop/ACIE/outputs/checkpoints/

# Or download all checkpoints
rsync -avz --progress \
  username@gpu-server.university.edu:~/acie/outputs/ \
  /Users/jitterx/Desktop/ACIE/outputs/

# ============================================================================
# ALTERNATIVE: SUBMIT TO JOB SCHEDULER (if cluster uses SLURM/PBS)
# ============================================================================

# Create SLURM job script
cat > train_job.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=acie_training
#SBATCH --output=logs/acie_%j.out
#SBATCH --error=logs/acie_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load modules (if needed)
module load cuda/11.8
module load python/3.9

# Activate environment
source ~/acie/venv/bin/activate

# Run training
cd ~/acie
python3 scripts/train_simple.py
EOF

# Submit job
sbatch train_job.slurm

# Check job status
squeue -u $USER

# View job output
tail -f logs/acie_*.out

# ============================================================================
# QUICK REFERENCE COMMANDS
# ============================================================================

# Sync code updates while training is running:
# rsync -avz /Users/jitterx/Desktop/ACIE/scripts/ username@server:~/acie/scripts/

# Kill training process:
# ssh username@server "kill $(cat ~/acie/training.pid)"

# Check disk space:
# ssh username@server "df -h ~/acie"

# Monitor with htop:
# ssh username@server "htop -u $USER"

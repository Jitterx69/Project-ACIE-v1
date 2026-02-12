# Remote Training Guide (RTX 4060)

This guide explains how to train the ACIE model on your remote laptop with an NVIDIA RTX 4060 GPU.

## Prerequisites
1.  **Remote Access**: SSH access to your remote laptop.
    -   Example: `ssh user@192.168.1.100`
2.  **Dataset**: Ensure your CSV datasets are on the remote machine.
    -   Recommended location: `~/acie_remote/data/`
3.  **NVIDIA Drivers**: Ensure drivers are installed on the remote machine (`nvidia-smi` works).

## Step 1: Sync Code
On your **local** machine (where you are now), run:

```bash
chmod +x scripts/sync_to_remote.sh
./scripts/sync_to_remote.sh user@192.168.1.100
```
(Replace `user@192.168.1.100` with your actual remote details)

## Step 2: Setup Remote Environment
SSH into your remote machine:

```bash
ssh user@192.168.1.100
cd ~/acie_remote
chmod +x scripts/setup_remote.sh
./scripts/setup_remote.sh
```

This will create a virtual environment `venv_remote` and install all dependencies optimized for CUDA.

## Step 3: Run Training
Activate the environment and start training:

```bash
source venv_remote/bin/activate

# Example: Train on 10k Observational Dataset
python acie/training/train.py \
    --data_dir data/ \
    --output_dir outputs/run1 \
    --dataset_size 10k \
    --gpus 1 \
    --batch_size 64 \
    --max_epochs 50
```

## Step 4: Monitoring (TensorBoard)
To view training progress locally, use SSH tunneling:

1.  **On Remote**: 
    ```bash
    tensorboard --logdir outputs/run1/logs --port 6006
    ```
2.  **On Local**:
    ```bash
    ssh -L 6006:localhost:6006 user@192.168.1.100
    ```
3.  **Open Browser**: Go to `http://localhost:6006`

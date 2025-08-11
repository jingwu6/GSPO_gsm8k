#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gspo

# Clear previous log
> nohup.out

# Kill existing python processes
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Fix pyarrow compatibility issue
pip install --force-reinstall pyarrow

# Start reference server in background
CUDA_VISIBLE_DEVICES=7 python ref_server.py &

# Wait a moment for ref server to start
sleep 5

# Start GSPO training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup deepspeed GSPO.py > nohup.out 2>&1 &

echo "Training started. Monitor with: tail -f nohup.out"
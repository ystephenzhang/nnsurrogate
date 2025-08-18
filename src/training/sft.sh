#!/bin/bash
# SFT training script for multiple models
# Usage: ./sft.sh

# Array of model configurations for training
# Format: "model_repo:model_name:output_suffix"
models=(
    "meta-llama/Llama-3.2-3B-Instruct:llama3.2-3b"
    "meta-llama/Llama-3.2-1B-Instruct:llama3.2-1b"
)

# Common training parameters
DATASET_PATH="/home/ubuntu/dev/data/SFT/heat_burgers_zero_shot_v1"
BASE_OUTPUT_DIR="/data/expo_output"
CUDA_DEVICES="0,1"

# Loop through each model configuration
for model_config in "${models[@]}"; do
    # Split the configuration into components
    IFS=':' read -r model_repo model_name <<< "$model_config"
    
    echo "Starting SFT training for: $model_name"
    echo "Model repository: $model_repo"
    
    # Set output directory for this model
    output_dir="$BASE_OUTPUT_DIR/${model_name}_heat_burgers_zero_shot"
    
    # Build and execute the training command
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    swift sft \
        --model $model_repo \
        --use_hf \
        --train_type lora \
        --output_dir $output_dir \
        --dataset $DATASET_PATH/train/data.json \
        --val_dataset $DATASET_PATH/val/data.json \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 8 \
        --learning_rate 1e-4 \
        --target_modules all-linear \
        --gradient_accumulation_steps 16 \
        --eval_steps 50 \
        --save_steps 50 \
        --save_total_limit 2 \
        --logging_steps 5 \
        --max_length 2048 \
        --system 'You are a helpful assistant with physics knowledge.' \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --model_author zy \
        --model_name ${model_name}-heat_burgers_zero_shot-sft
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed SFT training for $model_name"
        echo "Model saved to: $output_dir"
    else
        echo "✗ Failed SFT training for $model_name"
    fi
    echo "=========================================="
done

echo "All SFT training runs completed."
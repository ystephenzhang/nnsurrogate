set -eE -o pipefail  # Exit immediately on any command or pipeline error

# Parse GPU argument
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "🚀 Starting Llama-3.2-3B-Lora evaluation on GPU $GPU_ID"

# Change to SimulCost-Bench directory
cd /home/ubuntu/dev/SimulCost-Bench
export PYTHONPATH="/home/ubuntu/dev/SimulCost-Bench":$PYTHONPATH

RESUME_LOG="scripts/rerun_progress.log"
touch "$RESUME_LOG"

#1
cat > .env << 'EOF'
custom_code="/home/ubuntu/dev/SimulCost-Bench/custom_model/custom_inference.py"
model_path="/data/expo_output/llama3.2-3b_heat_burgers_zero_shot/v2-20250809-031426/checkpoint-22/"
custom_class="Llama3_lora_3B"
EOF

MODEL_NAME="llama3_2_3b_lora"
MODEL_DISPLAY_NAME="Llama-3.2-3B LoRA"

python inference/langchain_LLM.py -n 100 -p custom_model -m $MODEL_NAME -d 1D_heat_transfer -t cfl -z
python inference/langchain_LLM.py -n 100 -p custom_model -m $MODEL_NAME -d 1D_heat_transfer -t n_space -z

python evaluation/heat_transfer/eval.py -m $MODEL_NAME -d 1D_heat_transfer -t cfl -z
python evaluation/heat_transfer/eval.py -m $MODEL_NAME -d 1D_heat_transfer -t n_space -z

python inference/langchain_LLM.py -n 100 -p custom_model -m $MODEL_NAME -d 1D_heat_transfer -t cfl 
python inference/langchain_LLM.py -n 100 -p custom_model -m $MODEL_NAME -d 1D_heat_transfer -t n_space

python evaluation/heat_transfer/eval.py -m $MODEL_NAME -d 1D_heat_transfer -t cfl 
python evaluation/heat_transfer/eval.py -m $MODEL_NAME -d 1D_heat_transfer -t n_space
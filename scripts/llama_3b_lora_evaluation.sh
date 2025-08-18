#!/bin/bash
# llama_3b_lora_evaluation.sh
# Evaluation script for Llama-3.2-3B LoRA model
# Model path: /data/expo_output/llama3.2-3b_heat_burgers_zero_shot/v0-20250806-230008/
# Covers Heat_1D, Heat_steady_2D, and Burgers_1D tasks in both zero-shot and iterative settings
# Usage: ./llama_3b_lora_evaluation.sh [GPU_ID]

set -eE -o pipefail  # Exit immediately on any command or pipeline error

# Parse GPU argument
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "🚀 Starting Llama-3.2-3B LoRA evaluation on GPU $GPU_ID"

# Change to SimulCost-Bench directory
cd /home/ubuntu/dev/SimulCost-Bench
export PYTHONPATH="/home/ubuntu/dev/SimulCost-Bench":$PYTHONPATH

RESUME_LOG="scripts/llama_3b_lora_evaluation_progress.log"
touch "$RESUME_LOG"

# Function to run commands with resume capability
run_cmd () {
  local cmd="$1"
  # Skip if command is already logged
  if grep -Fxq "$cmd" "$RESUME_LOG"; then
    echo "✔ Already completed, skipping: $cmd"
    return
  fi

  echo "▶ Executing: $cmd"
  eval "$cmd"
  
  # Only append to log if successful
  echo "$cmd" >> "$RESUME_LOG"
}

# Update .env configuration for Llama-3.2-3B LoRA model
echo "🔧 Updating .env configuration for Llama-3.2-3B LoRA..."

#cat > .env << 'EOF'
## Custom Model Configuration
#custom_code="/home/ubuntu/dev/SimulCost-Bench/custom_model/custom_inference.py"
#model_path="/data/expo_output/llama3.2-3b_heat_burgers_zero_shot/v2-20250809-031426/checkpoint-22/"
#custom_class="Llama3_lora_3B"
#EOF

MODEL_NAME="llama3.2_3b_lora"
MODEL_DISPLAY_NAME="Llama-3.2-3B LoRA"

echo ""
echo "🚀 Starting evaluation for: $MODEL_DISPLAY_NAME"
echo "📊 Tasks: Heat_1D, Heat_steady_2D, Burgers_1D"
echo "🎯 Modes: Zero-shot (-z) and Iterative (no flag)"
echo "🖥️ GPU: $GPU_ID"
echo ""

if false;then
  # ========================================
  # HEAT_1D Tasks (1D Heat Transfer)
  # ========================================
  echo "🔥 Starting Heat_1D evaluations for $MODEL_DISPLAY_NAME..."

  heat_1d_tasks=("cfl" "n_space")
  modes=("-z" "")  # Zero-shot and iterative

  for mode in "${modes[@]}"; do
    mode_name=$([ "$mode" == "-z" ] && echo "zero-shot" || echo "iterative")
    echo "📋 Processing Heat_1D tasks in $mode_name mode..."
    
    for task in "${heat_1d_tasks[@]}"; do
      echo "🔧 Processing Heat_1D task: $task ($mode_name) - $MODEL_DISPLAY_NAME"
      
      # Run inference
      # Run inference
      #run_cmd "python inference/langchain_LLM.py -n 100 -p custom_model -m $MODEL_NAME -d 1D_heat_transfer -t $task $mode --resume"
      run_cmd "python inference/langchain_LLM.py -n 100 -p custom_model -m $MODEL_NAME -d 1D_heat_transfer -t $task $mode"
      
      # Run evaluation
      run_cmd "python evaluation/heat_transfer/eval.py -m $MODEL_NAME -d 1D_heat_transfer -t $task $mode"
    done
  done

  echo "✅ Heat_1D evaluations completed for $MODEL_DISPLAY_NAME!"
fi
# ========================================
# HEAT_STEADY_2D Tasks (2D Heat Transfer)
# ========================================
echo "🔥 Starting Heat_steady_2D evaluations for $MODEL_DISPLAY_NAME..."

# Define tasks and their supported modes
declare -A heat_2d_task_modes
heat_2d_task_modes["dx"]="-z "           # dx supports zero-shot and iterative
heat_2d_task_modes["error_threshold"]="-z "  # error_threshold supports zero-shot and iterative
#heat_2d_task_modes["dx"]="-z "           # dx supports zero-shot and iterative
#heat_2d_task_modes["error_threshold"]="-z "  # error_threshold supports zero-shot and iterative  
heat_2d_task_modes["relax"]="-z"         # relax only supports zero-shot
heat_2d_task_modes["t_init"]="-z"        # t_init only supports zero-shot

for task in "${!heat_2d_task_modes[@]}"; do
  modes_str="${heat_2d_task_modes[$task]}"
  
  # Determine supported modes for this task
  if [[ "$modes_str" == *" "* ]]; then
    # Contains space, supports both zero-shot and iterative
    task_modes=("-z" "")
  else
    # Only supports zero-shot
    task_modes=("$modes_str")
  fi
  
  for mode in "${task_modes[@]}"; do
    mode_name=$([ "$mode" == "-z" ] && echo "zero-shot" || echo "iterative")
    echo "🔧 Processing Heat_steady_2D task: $task ($mode_name) - $MODEL_DISPLAY_NAME"
    
    # Run inference
    # Run inference
    #run_cmd "python inference/langchain_LLM.py -n 30 -p custom_model -m $MODEL_NAME -d 2D_heat_transfer -t $task $mode --resume"
    run_cmd "python inference/langchain_LLM.py -n 50 -p custom_model -m $MODEL_NAME -d 2D_heat_transfer -t $task $mode"
    
    # Run evaluation
    run_cmd "python evaluation/heat_transfer/eval.py -m $MODEL_NAME -d 2D_heat_transfer -t $task $mode"
  done
done

echo "✅ Heat_steady_2D evaluations completed for $MODEL_DISPLAY_NAME!"

# ========================================
# BURGERS_1D Tasks (1D Burgers Equation)
# ========================================
echo "🌊 Starting Burgers_1D evaluations for $MODEL_DISPLAY_NAME..."

burgers_1d_tasks=("cfl" "k" "w")
#burgers_1d_cases=("blast" "double_shock" "rarefaction" "sin" "sod")
burgers_1d_cases=("blast" "double_shock" "rarefaction")
#modes=("-z" "")  # Zero-shot and iterative
modes=("")  # Iterative

for mode in "${modes[@]}"; do
  mode_name=$([ "$mode" == "-z" ] && echo "zero-shot" || echo "iterative")
  echo "📋 Processing Burgers_1D tasks in $mode_name mode..."
  
  for task in "${burgers_1d_tasks[@]}"; do
    for case in "${burgers_1d_cases[@]}"; do
      echo "🔧 Processing Burgers_1D task: $task, case: $case ($mode_name) - $MODEL_DISPLAY_NAME"
      
      # Run inference
      # Run inference
      #run_cmd "python inference/langchain_LLM.py -p custom_model -m $MODEL_NAME -d burgers_1d -t $task -c $case $mode --resume"
      run_cmd "python inference/langchain_LLM.py -p custom_model -m $MODEL_NAME -d burgers_1d -t $task -c $case $mode"
      
      # Run evaluation
      run_cmd "python evaluation/burgers/eval.py -m $MODEL_NAME -d burgers_1d -t $task -c $case $mode"
    done
  done
done

echo "✅ Burgers_1D evaluations completed for $MODEL_DISPLAY_NAME!"

# ========================================
# TABULATION
# ========================================
echo "=========================================="
echo "GENERATING RESULT TABULATIONS"
echo "=========================================="

echo "📊 Generating result tabulations for all datasets..."

# Generate tabulation for each dataset
datasets=("1D_heat_transfer" "2D_heat_transfer" "burgers_1d")

for dataset in "${datasets[@]}"; do
  echo "📈 Tabulating results for: $dataset"
  run_cmd "python evaluation/tabulate.py -d $dataset"
done

# ========================================
# COMPLETION SUMMARY
# ========================================
echo ""
echo "🎉 LLAMA-3.2-3B LORA EVALUATION COMPLETED SUCCESSFULLY!"
echo ""
echo "🔍 Results can be found in:"
echo "   - Individual results: results_model_attempt/{dataset}/{task}/"
echo "   - Evaluation results: eval_results/{dataset}/{task}/"
echo "   - Logs: log_model_tool_call/{dataset}/{task}/"
echo "   - Tabulated summaries: Generated by tabulate.py"
echo ""
echo "📋 Summary of completed evaluations:"
echo "   🤖 Model: $MODEL_DISPLAY_NAME"
echo "   🖥️ GPU: $GPU_ID"
echo ""
echo "   🔥 Heat_1D: 2 tasks × 2 modes = 4 evaluations"
echo "   🔥 Heat_steady_2D: 4 tasks (2 with both modes, 2 zero-shot only) = 6 evaluations"  
echo "   🌊 Burgers_1D: 3 tasks × 5 cases × 2 modes = 30 evaluations"
echo "   📊 Total: 40 individual evaluations"
echo ""
echo "🚀 Llama-3.2-3B LoRA evaluation complete!"
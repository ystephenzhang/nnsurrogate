#!/bin/bash
# llama_additional_models_evaluation.sh
# Comprehensive evaluation script for additional Llama models:
# 1. meta-llama/Llama-3.2-1B-Instruct (base model)
# 2. Llama-3.2-1B LoRA model (/data/expo_output/llama3.2-1b_heat_burgers_zero_shot/)
# 3. Llama-3.2-3B LoRA model (/data/expo_output/llama3.2-3b_heat_burgers_zero_shot/)
# Covers Heat_1D, Heat_steady_2D, and Burgers_1D tasks in both zero-shot and iterative settings

set -eE -o pipefail  # Exit immediately on any command or pipeline error

# Change to SimulCost-Bench directory
cd /home/ubuntu/dev/SimulCost-Bench
export PYTHONPATH="/home/ubuntu/dev/SimulCost-Bench":$PYTHONPATH

RESUME_LOG="scripts/llama_additional_models_evaluation_progress.log"
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

# Function to update .env file for different models
update_env_config() {
  local model_path="$1"
  local custom_class="$2"
  
  echo "🔧 Updating .env configuration:"
  echo "   Model path: $model_path"
  echo "   Custom class: $custom_class"
  
  cat > .env << EOF
# Custom Model Configuration
custom_code="/home/ubuntu/dev/SimulCost-Bench/custom_model/llama_inference.py"
model_path="$model_path"
custom_class="$custom_class"
EOF
}

# Function to run full evaluation for a single model
run_full_evaluation() {
  local model_name="$1"
  local model_display_name="$2"
  
  echo ""
  echo "🚀 Starting evaluation for: $model_display_name"
  echo "📊 Tasks: Heat_1D, Heat_steady_2D, Burgers_1D"
  echo "🎯 Modes: Zero-shot (-z) and Iterative (no flag)"
  echo ""
  
  # ========================================
  # HEAT_1D Tasks (1D Heat Transfer)
  # ========================================
  echo "🔥 Starting Heat_1D evaluations for $model_display_name..."

  heat_1d_tasks=("cfl" "n_space")
  modes=("-z" "")  # Zero-shot and iterative

  for mode in "${modes[@]}"; do
    mode_name=$([ "$mode" == "-z" ] && echo "zero-shot" || echo "iterative")
    echo "📋 Processing Heat_1D tasks in $mode_name mode..."
    
    for task in "${heat_1d_tasks[@]}"; do
      echo "🔧 Processing Heat_1D task: $task ($mode_name) - $model_display_name"
      
      # Run inference
      run_cmd "python inference/langchain_LLM.py -n 100 -p custom_model -m $model_name -d 1D_heat_transfer -t $task $mode --resume"
      
      # Run evaluation
      run_cmd "python evaluation/heat_transfer/eval.py -m $model_name -d 1D_heat_transfer -t $task $mode"
    done
  done

  echo "✅ Heat_1D evaluations completed for $model_display_name!"

  # ========================================
  # HEAT_STEADY_2D Tasks (2D Heat Transfer)
  # ========================================
  echo "🔥 Starting Heat_steady_2D evaluations for $model_display_name..."

  # Define tasks and their supported modes
  declare -A heat_2d_task_modes
  heat_2d_task_modes["dx"]="-z "           # dx supports zero-shot and iterative
  heat_2d_task_modes["error_threshold"]="-z "  # error_threshold supports zero-shot and iterative  
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
      echo "🔧 Processing Heat_steady_2D task: $task ($mode_name) - $model_display_name"
      
      # Run inference
      run_cmd "python inference/langchain_LLM.py -n 100 -p custom_model -m $model_name -d 2D_heat_transfer -t $task $mode --resume"
      
      # Run evaluation
      run_cmd "python evaluation/heat_transfer/eval.py -m $model_name -d 2D_heat_transfer -t $task $mode"
    done
  done

  echo "✅ Heat_steady_2D evaluations completed for $model_display_name!"

  # ========================================
  # BURGERS_1D Tasks (1D Burgers Equation)
  # ========================================
  echo "🌊 Starting Burgers_1D evaluations for $model_display_name..."

  burgers_1d_tasks=("cfl" "k" "w")
  burgers_1d_cases=("blast" "double_shock" "rarefaction" "sin" "sod")
  modes=("-z" "")  # Zero-shot and iterative

  for mode in "${modes[@]}"; do
    mode_name=$([ "$mode" == "-z" ] && echo "zero-shot" || echo "iterative")
    echo "📋 Processing Burgers_1D tasks in $mode_name mode..."
    
    for task in "${burgers_1d_tasks[@]}"; do
      for case in "${burgers_1d_cases[@]}"; do
        echo "🔧 Processing Burgers_1D task: $task, case: $case ($mode_name) - $model_display_name"
        
        # Run inference
        run_cmd "python inference/langchain_LLM.py -p custom_model -m $model_name -d burgers_1d -t $task -c $case $mode --resume"
        
        # Run evaluation
        run_cmd "python evaluation/burgers/eval.py -m $model_name -d burgers_1d -t $task -c $case $mode"
      done
    done
  done

  echo "✅ Burgers_1D evaluations completed for $model_display_name!"
  echo "🎉 All evaluations completed for $model_display_name!"
}

# ========================================
# MAIN EXECUTION - EVALUATE ALL MODELS
# ========================================

echo "🚀 Starting comprehensive evaluation of additional Llama models"
echo "📋 Models to evaluate:"
echo "   1. Llama-3.2-1B-Instruct (base model)"
echo "   2. Llama-3.2-1B LoRA (/data/expo_output/llama3.2-1b_heat_burgers_zero_shot/)"
echo "   3. Llama-3.2-3B LoRA (/data/expo_output/llama3.2-3b_heat_burgers_zero_shot/)"
echo ""

# ========================================
# MODEL 1: Llama-3.2-1B-Instruct (Base)
# ========================================
echo "=========================================="
echo "MODEL 1: Llama-3.2-1B-Instruct (Base)"
echo "=========================================="

update_env_config "meta-llama/Llama-3.2-1B-Instruct" "Llama3"
run_full_evaluation "llama3_2_1b" "Llama-3.2-1B-Instruct (Base)"

# ========================================
# MODEL 2: Llama-3.2-1B LoRA
# ========================================
echo "=========================================="
echo "MODEL 2: Llama-3.2-1B LoRA"
echo "=========================================="

update_env_config "/data/expo_output/llama3.2-1b_heat_burgers_zero_shot/v0-20250806-230427/" "Llama3_lora"
run_full_evaluation "llama3_2_1b_lora" "Llama-3.2-1B LoRA"

# ========================================
# MODEL 3: Llama-3.2-3B LoRA
# ========================================
echo "=========================================="
echo "MODEL 3: Llama-3.2-3B LoRA"
echo "=========================================="

update_env_config "/data/expo_output/llama3.2-3b_heat_burgers_zero_shot/v0-20250806-230008/" "Llama3_lora"
run_full_evaluation "llama3_2_3b_lora" "Llama-3.2-3B LoRA"

# ========================================
# FINAL TABULATION
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
echo "🎉 ALL ADDITIONAL MODEL EVALUATIONS COMPLETED SUCCESSFULLY!"
echo ""
echo "🔍 Results can be found in:"
echo "   - Individual results: results_model_attempt/{dataset}/{task}/"
echo "   - Evaluation results: eval_results/{dataset}/{task}/"
echo "   - Logs: log_model_tool_call/{dataset}/{task}/"
echo "   - Tabulated summaries: Generated by tabulate.py"
echo ""
echo "📋 Summary of completed evaluations:"
echo "   🤖 Models evaluated: 3"
echo "     1. llama3_2_1b (Llama-3.2-1B-Instruct base)"
echo "     2. llama3_2_1b_lora (Llama-3.2-1B LoRA fine-tuned)"
echo "     3. llama3_2_3b_lora (Llama-3.2-3B LoRA fine-tuned)"
echo ""
echo "   🔥 Heat_1D: 2 tasks × 2 modes = 4 evaluations per model"
echo "   🔥 Heat_steady_2D: 4 tasks (2 with both modes, 2 zero-shot only) = 6 evaluations per model"  
echo "   🌊 Burgers_1D: 3 tasks × 5 cases × 2 modes = 30 evaluations per model"
echo "   📊 Per model: 40 individual evaluations"
echo "   📊 Total: 120 individual model evaluations"
echo ""
echo "🚀 Additional Llama model evaluations complete!"

# Reset .env to previous configuration (3B base model)
echo "🔧 Restoring .env to Llama-3.2-3B-Instruct configuration..."
update_env_config "meta-llama/Llama-3.2-3B-Instruct" "Llama3"
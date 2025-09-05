#!/bin/bash

# Mass Evaluation Template for SimulCost-Bench
# Supports Heat_1d n_space and Euler_1d cfl tasks at high precision
# Usage: ./eval_template.sh <model_name> <problem> <mode> [backend]
#   model_name: Name of the model to evaluate
#   problem: Either "heat_1d_n_space" or "euler_1d_cfl"
#   mode: Either "zero-shot" or "iterative"
#   backend: Either "openai" or "custom_model" (defaults to "custom_model")

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 <model_name> <problem> <mode> [backend]"
    echo ""
    echo "Arguments:"
    echo "  model_name    Name of the model to evaluate (e.g., 'gpt-4o', 'qwen3_8b')"
    echo "  problem       Either 'heat_1d_n_space' or 'euler_1d_cfl'"
    echo "  mode          Either 'zero-shot' or 'iterative'"
    echo "  backend       Either 'openai' or 'custom_model' (default: 'custom_model')"
    echo ""
    echo "Examples:"
    echo "  $0 gpt-4o heat_1d_n_space zero-shot openai"
    echo "  $0 qwen3_8b euler_1d_cfl iterative custom_model"
    echo "  $0 iterative_baseline heat_1d_n_space zero-shot"  # Uses default backend 'custom_model'
    exit 1
}



MODEL_NAME="$1"
PROBLEM="$2"
MODE="$3"
TASK="$4"
PRECISION="$5"
BACKEND="${6:-custom_model}"  # Default to custom_model if not specified

# Validate backend
if [[ "$BACKEND" != "openai" && "$BACKEND" != "custom_model" ]]; then
    echo "Error: Backend must be either 'openai' or 'custom_model'"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "zero-shot" && "$MODE" != "iterative" ]]; then
    echo "Error: Mode must be either 'zero-shot' or 'iterative'"
    exit 1
fi

# Parse problem and set evaluation parameters
case "$PROBLEM" in
    "heat_1d")
        DATASET="heat_1d"
        #TASK="n_space"
        EVAL_SCRIPT="evaluation/heat_1d/eval.py"
        #PRECISION="high"
        ;;
    "euler_1d")
        DATASET="euler_1d"
        #TASK="n_space"
        EVAL_SCRIPT="evaluation/euler_1d/eval.py"
        #PRECISION="high"
        ;;
    *)
        echo "Error: Problem must be either 'heat_1d_n_space' or 'euler_1d_cfl'"
        exit 1
        ;;
esac

# Fixed parameters for your requirements

# Set mode flags based on input
if [[ "$MODE" == "zero-shot" ]]; then
    MODE_FLAG="-z"
    MODE_DISPLAY="Zero-shot"
else
    MODE_FLAG=""  # No flag for iterative mode
    MODE_DISPLAY="Iterative"
fi

echo "=== Starting Mass Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Backend: $BACKEND"
echo "Dataset: $DATASET"
echo "Task: $TASK"
echo "Precision: $PRECISION"
echo "Mode: $MODE_DISPLAY"
echo "================================"

# Start timing
START_TIME=$(date +%s)

# Change to SimulCost-Bench directory
cd /home/ubuntu/dev/SimulCost-Bench

# Run inference
echo "Running inference..."
if [[ -n "$MODE_FLAG" ]]; then
    python inference/langchain_LLM.py \
        -p "$BACKEND" \
        -m "$MODEL_NAME" \
        -d "$DATASET" \
        -t "$TASK" \
        -l "$PRECISION" \
        $MODE_FLAG  
else
    python inference/langchain_LLM.py \
        -p "$BACKEND" \
        -m "$MODEL_NAME" \
        -d "$DATASET" \
        -t "$TASK" \
        -l "$PRECISION" \
        --resume
fi

# Check if inference succeeded
if [ $? -ne 0 ]; then
    echo "Error: Inference failed"
    exit 1
fi

echo "Inference completed successfully"

# Run evaluation
echo "Running evaluation..."
if [[ -n "$MODE_FLAG" ]]; then
    python "$EVAL_SCRIPT" \
        -m "$MODEL_NAME" \
        -t "$TASK" \
        -l "$PRECISION" \
        $MODE_FLAG
else
    python "$EVAL_SCRIPT" \
        -m "$MODEL_NAME" \
        -t "$TASK" \
        -l "$PRECISION"
fi

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo "Evaluation completed successfully"

# Calculate and log runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create runtime log entry
RUNTIME_LOG_FILE="/home/ubuntu/dev/SimulCost-Bench/eval_results/runtime_log/runtime_${MODEL_NAME}_${DATASET}_${TASK}_${PRECISION}_${MODE}.log"
echo "=== Runtime Log ===" > "$RUNTIME_LOG_FILE"
echo "Timestamp: $TIMESTAMP" >> "$RUNTIME_LOG_FILE"
echo "Model: $MODEL_NAME" >> "$RUNTIME_LOG_FILE"
echo "Backend: $BACKEND" >> "$RUNTIME_LOG_FILE"
echo "Dataset: $DATASET" >> "$RUNTIME_LOG_FILE"
echo "Task: $TASK" >> "$RUNTIME_LOG_FILE"
echo "Precision: $PRECISION" >> "$RUNTIME_LOG_FILE"
echo "Mode: $MODE_DISPLAY" >> "$RUNTIME_LOG_FILE"
echo "Total Runtime: ${RUNTIME} seconds" >> "$RUNTIME_LOG_FILE"
echo "===================" >> "$RUNTIME_LOG_FILE"

# Return to original directory
cd - > /dev/null

echo "=== Mass Evaluation Complete ==="
echo "Results saved in SimulCost-Bench/eval_results/$DATASET/$TASK/$PRECISION/"
echo "Runtime: ${RUNTIME} seconds (logged to $RUNTIME_LOG_FILE)"

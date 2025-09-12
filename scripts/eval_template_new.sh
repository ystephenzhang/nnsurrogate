#!/bin/bash

# Mass Evaluation Template for SimulCost-Bench
# Automatically maps problem to task: heat_1d->n_space, euler_1d->n_space, ns_channel_2d->mesh_x
# Usage: ./eval_template_new.sh <problem> <model_name> <mode> <backend> [precision]
#   problem: Either "heat_1d", "euler_1d", or "ns_channel_2d"
#   model_name: Name of the model to evaluate
#   mode: Either "zero-shot" or "iterative"  
#   backend: Either "openai" or "custom_model"
#   precision: Either "low", "medium", "high" (optional, if not specified runs all three)

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 <problem> <model_name> <mode> <backend> [precision] [resume]"
    echo ""
    echo "Arguments:"
    echo "  problem       Either 'heat_1d', 'euler_1d', or 'ns_channel_2d'"
    echo "  model_name    Name of the model to evaluate (e.g., 'gpt-4o', 'qwen3_8b')"
    echo "  mode          Either 'zero-shot' or 'iterative'"
    echo "  backend       Either 'openai' or 'custom_model'"
    echo "  precision     Either 'low', 'medium', 'high' (optional, runs all if not specified)"
    echo "  resume        Either 'resume' or 'no-resume' (optional, default: 'no-resume' for all modes)"
    echo ""
    echo "Task mapping:"
    echo "  heat_1d      -> n_space"
    echo "  euler_1d     -> n_space" 
    echo "  ns_channel_2d -> mesh_x"
    echo ""
    echo "Examples:"
    echo "  $0 heat_1d gpt-4o zero-shot openai high"
    echo "  $0 euler_1d qwen3_8b iterative custom_model medium no-resume"  # Force no resume"
    echo "  $0 ns_channel_2d claude-3-haiku zero-shot custom_model medium resume"  # Force resume"
    exit 1
}

# Check minimum arguments
if [ $# -lt 4 ]; then
    usage
fi

PROBLEM="$1"
MODEL_NAME="$2"
MODE="$3"
BACKEND="$4"
PRECISION="${5:-all}"  # Default to all if not specified
RESUME_ARG="${6:-auto}"  # Default to auto (will be determined by mode)

# Validate precision if specified
if [[ "$PRECISION" != "all" && "$PRECISION" != "low" && "$PRECISION" != "medium" && "$PRECISION" != "high" ]]; then
    echo "Error: Precision must be either 'low', 'medium', 'high', or omitted for all"
    exit 1
fi

# Validate resume argument if specified
if [[ "$RESUME_ARG" != "auto" && "$RESUME_ARG" != "resume" && "$RESUME_ARG" != "no-resume" ]]; then
    echo "Error: Resume argument must be either 'resume', 'no-resume', or omitted for auto"
    exit 1
fi

# Parse problem and set evaluation parameters with automatic task mapping
case "$PROBLEM" in
    "heat_1d")
        DATASET="heat_1d"
        TASK="n_space"
        EVAL_SCRIPT="evaluation/heat_1d/eval.py"
        ;;
    "euler_1d")
        DATASET="euler_1d"
        TASK="n_space"
        EVAL_SCRIPT="evaluation/euler_1d/eval.py"
        ;;
    "ns_2d")
        DATASET="ns_2d"
        TASK="mesh_x"
        EVAL_SCRIPT="evaluation/ns_2d/eval.py"
        ;;
    "ns_transient_2d")
        DATASET="ns_transient_2d"
        TASK="resolution"
        EVAL_SCRIPT="evaluation/ns_transient_2d/eval.py"
        ;;
    *)
        echo "Error: Problem must be either 'heat_1d', 'euler_1d', or 'ns_2d'"
        exit 1
        ;;
esac

# Determine resume behavior
if [[ "$RESUME_ARG" == "auto" ]]; then
    # Default behavior: no resume for all modes
    USE_RESUME=false
elif [[ "$RESUME_ARG" == "resume" ]]; then
    USE_RESUME=true
else
    USE_RESUME=false
fi

# Set mode flags based on input
if [[ "$MODE" == "zero-shot" ]]; then
    MODE_FLAG="-z"
    MODE_DISPLAY="Zero-shot"
else
    MODE_FLAG=""  # No flag for iterative mode
    MODE_DISPLAY="Iterative"
fi

# Set resume display
if [[ "$USE_RESUME" == true ]]; then
    RESUME_DISPLAY="with resume"
else
    RESUME_DISPLAY="without resume"
fi

# Function to run evaluation for a single precision level
run_single_precision() {
    local prec=$1
    
    echo "=== Running Evaluation for Precision: $prec ==="
    echo "Model: $MODEL_NAME"
    echo "Backend: $BACKEND"
    echo "Dataset: $DATASET"
    echo "Task: $TASK"
    echo "Precision: $prec"
    echo "Mode: $MODE_DISPLAY $RESUME_DISPLAY"
    echo "================================"

    # Change to SimulCost-Bench directory
    cd /home/ubuntu/dev/SimulCost-Bench

    # Run inference
    echo "Running inference for precision $prec..."
    if [[ -n "$MODE_FLAG" ]]; then
        # Zero-shot mode
        if [[ "$USE_RESUME" == true ]]; then
            python inference/langchain_LLM.py \
                -p "$BACKEND" \
                -m "$MODEL_NAME" \
                -d "$DATASET" \
                -t "$TASK" \
                -l "$prec" \
                $MODE_FLAG \
                --resume
        else
            python inference/langchain_LLM.py \
                -p "$BACKEND" \
                -m "$MODEL_NAME" \
                -d "$DATASET" \
                -t "$TASK" \
                -l "$prec" \
                $MODE_FLAG
        fi
    else
        # Iterative mode
        if [[ "$USE_RESUME" == true ]]; then
            python inference/langchain_LLM.py \
                -p "$BACKEND" \
                -m "$MODEL_NAME" \
                -d "$DATASET" \
                -t "$TASK" \
                -l "$prec" \
                --resume
        else
            python inference/langchain_LLM.py \
                -p "$BACKEND" \
                -m "$MODEL_NAME" \
                -d "$DATASET" \
                -t "$TASK" \
                -l "$prec"
        fi
    fi

    # Check if inference succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for precision $prec"
        return 1
    fi

    echo "Inference completed successfully for precision $prec"

    # Run evaluation
    echo "Running evaluation for precision $prec..."
    if [[ -n "$MODE_FLAG" ]]; then
        python "$EVAL_SCRIPT" \
            -m "$MODEL_NAME" \
            -t "$TASK" \
            -l "$prec" \
            $MODE_FLAG
    else
        python "$EVAL_SCRIPT" \
            -m "$MODEL_NAME" \
            -t "$TASK" \
            -l "$prec"
    fi

    # Check if evaluation succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for precision $prec"
        return 1
    fi

    echo "Evaluation completed successfully for precision $prec"
    
    # Return to original directory
    cd - > /dev/null
    
    return 0
}

# Start timing
START_TIME=$(date +%s)

# Run evaluation for specified precision level(s)
if [[ "$PRECISION" == "all" ]]; then
    echo "=== Starting Mass Evaluation for All Precision Levels ==="
    PRECISIONS=("low" "medium" "high")
    #PRECISIONS=("medium" "high")
    
    for prec in "${PRECISIONS[@]}"; do
        echo ""
        echo ">>> Processing precision level: $prec <<<"
        run_single_precision "$prec"
        if [ $? -ne 0 ]; then
            echo "Error: Failed at precision level $prec"
            exit 1
        fi
        echo ">>> Completed precision level: $prec <<<"
        echo ""
    done
    
    PRECISION_DISPLAY="all (low, medium, high)"
else
    echo "=== Starting Mass Evaluation for Single Precision Level ==="
    run_single_precision "$PRECISION"
    if [ $? -ne 0 ]; then
        exit 1
    fi
    PRECISION_DISPLAY="$PRECISION"
fi

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
echo "Precision: $PRECISION_DISPLAY" >> "$RUNTIME_LOG_FILE"
echo "Mode: $MODE_DISPLAY" >> "$RUNTIME_LOG_FILE"
echo "Total Runtime: ${RUNTIME} seconds" >> "$RUNTIME_LOG_FILE"
echo "===================" >> "$RUNTIME_LOG_FILE"

echo "=== Mass Evaluation Complete ==="
echo "Results saved in SimulCost-Bench/eval_results/$DATASET/$TASK/"
echo "Precision levels processed: $PRECISION_DISPLAY"
echo "Total runtime: ${RUNTIME} seconds (logged to $RUNTIME_LOG_FILE)"

import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import ast
from typing import List, Tuple, Union, Dict
from pathlib import Path

from optimization.simu_utils import *
spec = importlib.util.spec_from_file_location("src_utils", "/home/ubuntu/dev/src/general_utils.py")
src_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(src_utils)
Verifier = src_utils.Verifier

verifier = Verifier("euler_1d", "n_space", "medium")

def extract_optimization_data(filenames: List[str]) -> List[List[Tuple[int, float, int]]]:
    """
    Extract optimization data from JSON files.

    Args:
        filenames: List of JSON file paths

    Returns:
        List of lists, where each inner list contains (step, real_value, n_space) tuples
        for each file, selecting best based on predicted value
    """
    all_file_data = []

    for filename in filenames:
        with open(filename, 'r') as f:
            data = json.load(f)

        if 'old_value_pairs_with_i_step' not in data:
            print(f"Warning: 'old_value_pairs_with_i_step' not found in {filename}")
            continue

        file_step_data = []
        pairs = data['old_value_pairs_with_i_step']

        # Process each step
        for step_key in sorted(pairs.keys(), key=lambda x: int(x) if x != "-1" else -1):
            if step_key == "-1":
                continue

            step_num = int(step_key)
            step_entries = pairs[step_key]

            # Find best entry based on predicted value
            best_entry = None
            best_predicted = float('-inf')

            for entry in step_entries:
                if len(entry) >= 2:
                    # Extract predicted and real values
                    predicted_value = entry[1][-1]  # Last element of predicted values

                    if predicted_value > best_predicted:
                        best_predicted = predicted_value
                        best_entry = entry

            if best_entry is not None:
                # Get real value for the best predicted entry
                if len(best_entry) >= 3:
                    real_value = best_entry[2][-1]  # Last element of real values
                else:
                    #real_value = best_entry[1][-1]  # If only 2 items, real value is entry[1][-1]
                    
                    real_value = evaluate(best_entry[0],
                                        verifier,
                                        profile,
                                        qid,
                                        backend=self.evaluation_backend,
                                        surrogate_config=self.cfg.surrogate_config[problem][task][tolerance],
                                        use_soft_success=self.soft_success,
                                        prev_cost=prev_cost,
                                        threshold=self.threshold)

                # Extract n_space parameter from the params (first element)
                params = best_entry[0]
                n_space = params.get('n_space', 'N/A') if isinstance(params, dict) else 'N/A'

                file_step_data.append((step_num, real_value, n_space))

        all_file_data.append(file_step_data)

    return all_file_data


def plot_optimization_with_variance(filenames: Union[str, List[str]], labels: List[str] = None, success_threshold: float = None, base: float = None):
    """
    Plot optimization results showing real values for entries selected by best predicted values,
    with n_space annotations on the dots. Can handle single file or multiple files.

    Args:
        filenames: Single JSON file path (string) or list of JSON file paths
        labels: Optional labels for each file (defaults to filename)
        success_threshold: Optional success threshold value to draw as red dotted horizontal line
        base: Optional base model value to draw as horizontal line using the same style as other methods
    """
    # Convert single filename to list for uniform handling
    if isinstance(filenames, str):
        filenames = [filenames]

    if labels is None:
        labels = [Path(f).stem for f in filenames]
    elif len(labels) != len(filenames):
        print(f"Warning: Number of labels ({len(labels)}) doesn't match number of files ({len(filenames)})")
        labels = [Path(f).stem for f in filenames]

    # Extract data
    all_data = extract_optimization_data(filenames)

    plt.figure(figsize=(12, 8))

    # Use different colors and markers for each file
    # Reserve space for base line if provided
    total_lines = len(filenames) + (1 if base is not None else 0)
    colors = plt.cm.tab10(np.linspace(0, 1, total_lines))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (file_data, label) in enumerate(zip(all_data, labels)):
        if not file_data:
            print(f"Warning: No data found for {filenames[i]}")
            continue

        steps, real_values, n_spaces = zip(*file_data)

        # Plot line with different colors and markers
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(steps, real_values,
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=6)

        # Add n_space annotations on the dots with matching colors
        for step, real_val, n_space in file_data:
            plt.annotate(str(n_space),
                        (step, real_val),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8,
                        alpha=0.7,
                        color=color)

    # Add base model line if provided
    if base is not None:
        # Use the next color in the palette for the base line
        base_color = colors[len(filenames) % len(colors)]
        base_marker = markers[len(filenames) % len(markers)]

        # Get the x-axis range to draw the horizontal line across the entire plot
        if all_data and any(file_data for file_data in all_data):
            # Find the range of steps across all data
            all_steps = []
            for file_data in all_data:
                if file_data:
                    steps, _, _ = zip(*file_data)
                    all_steps.extend(steps)

            if all_steps:
                x_min, x_max = min(all_steps), max(all_steps)
                # Draw horizontal line with same style as other methods
                plt.plot([x_min, x_max], [base, base],
                        color=base_color,
                        linewidth=2,
                        label='Base Model',
                        linestyle='-')

    # Add success threshold line if provided
    if success_threshold is not None:
        plt.axhline(y=success_threshold, color='red', linestyle=':',
                   linewidth=2, alpha=0.8, label=f'Success Threshold ({success_threshold})')

    plt.xlabel('Optimization Step')
    plt.ylabel('Soft Success Score')
    plt.title('Zero-shot Optimization Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_optimization_results(names, base_path="/home/ubuntu/dev/outputs/optimization-results"):
    """
    Plot optimization results for multiple methods.

    Args:
        names: List of folder names under base_path
        base_path: Base directory containing optimization results
    """
    plt.figure(figsize=(10, 6))

    for name in names:
        folder_path = Path(base_path) / name

        # Find log files in the folder
        log_files = list(folder_path.glob("*.json"))

        if not log_files:
            print(f"No log files found in {folder_path}")
            continue

        # Use the first log file found
        log_file = log_files[0]

        with open(log_file, 'r') as f:
            data = json.load(f)

        if 'old_value_pairs_with_i_step' not in data:
            print(f"'old_value_pairs_with_i_step' key not found in {log_file}")
            continue

        steps = []
        best_real_values = []

        pairs = data['old_value_pairs_with_i_step']

        # Process each step
        for step in pairs:
            if step == "-1":
                continue
            points = pairs[step]

            ranked_points = sorted(points, key=lambda x: (-x[1][0], -x[1][1]))
            best_real_values.append(int(ranked_points[0][2][0] >= 1.0) * ranked_points[0][2][1])
            steps.append(step)

        # Plot the line for this method
        if steps and best_real_values:
            plt.plot(steps, best_real_values, marker='o', label=name, linewidth=2, markersize=6)
        else:
            print(f"No valid data points found for {name}")

    plt.xlabel('Step')
    plt.ylabel('Real Value')
    plt.title('Optimization Progress: Best Real Value per Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.show()


def extract_tool_call_data(file_paths: List[str]) -> List[List[Tuple[str, float, int]]]:
    """
    Extract tool call data from log files.

    Args:
        file_paths: List of log file paths

    Returns:
        List of lists, where each inner list contains (tool_args, RMSE, accumulated_cost) tuples for each QID
    """
    all_qid_data = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            content = f.read()

        # Detect file format and extract accordingly
        if '<� QID:' in content and '=� RMSE (model vs. dummy):' in content:
            # Evaluation results format
            qid_data = extract_evaluation_format(content)
        else:
            # Tool call log format
            qid_data = extract_tool_call_format(content)

        all_qid_data.extend(qid_data)

    return all_qid_data


def extract_evaluation_format(content: str) -> List[List[Tuple[str, float, int]]]:
    """Extract data from evaluation results format."""
    qid_blocks = re.split(r'=� --- Evaluation Result ---', content)
    qid_data = []

    for block in qid_blocks[1:]:  # Skip first empty block
        # Extract QID
        qid_match = re.search(r'<� QID:\s*(\d+)', block)
        if not qid_match:
            continue

        # Extract RMSE
        rmse_match = re.search(r'=� RMSE \(model vs\. dummy\):\s*([0-9.e-]+)', block)
        rmse = float(rmse_match.group(1)) if rmse_match else 0.0

        # Extract Model Cost
        cost_match = re.search(r'=� Model Cost:\s*(\d+)', block)
        cost = int(cost_match.group(1)) if cost_match else 0

        # Extract Model Parameters
        param_match = re.search(r'=� Model Parameters:\s*\{([^}]*)\}', block, re.DOTALL)
        if param_match:
            param_str = '{' + param_match.group(1) + '}'
            # Clean up the parameter string format
            param_str = re.sub(r'\s*"([^"]+)":\s*([^,\n}]+)', r'"\1": \2', param_str)
            tool_args = param_str
        else:
            tool_args = "N/A"

        # Each evaluation result represents a single "step" for this QID
        qid_data.append([(tool_args, rmse, cost)])

    return qid_data


def extract_tool_call_format(content: str) -> List[List[Tuple[str, float, int]]]:
    """Extract data from tool call log format."""
    qid_data = []

    # Split by QID sections
    qid_sections = re.split(r'========== >� The model begins to solve a new problem ==========', content)

    for section in qid_sections[1:]:  # Skip first empty section
        qid_tool_calls = []

        # Find all tool calls and their results in this QID using a more flexible approach
        # Split by tool call blocks
        tool_call_blocks = re.findall(
            r'QID=(\d+) - =� Received tool call: \{(.*?)\}(?=\n\[INFO|\nQID|\Z)',
            section,
            re.DOTALL
        )

        # Find RMSE and accumulated_cost from results
        result_blocks = re.findall(
            r'QID=\d+ -  Tool call result: \{(.*?)\}(?=\n\[INFO|\nQID|\Z)',
            section,
            re.DOTALL
        )

        # Process each tool call block
        for i, (qid, tool_call_content) in enumerate(tool_call_blocks):
            if i < len(result_blocks):
                result_content = result_blocks[i]

                # Extract RMSE
                rmse_match = re.search(r'"RMSE":\s*([0-9.]+)', result_content)
                rmse = float(rmse_match.group(1)) if rmse_match else 0.0

                # Extract accumulated_cost
                cost_match = re.search(r'"accumulated_cost":\s*([0-9]+)', result_content)
                accumulated_cost = int(cost_match.group(1)) if cost_match else 0

                # Extract tool_args - handle both string and JSON object formats
                tool_args = "N/A"  # Default value

                # Try to find tool_args as a string (format 1)
                string_match = re.search(r'"tool_args":\s*"([^"]*)"', tool_call_content)
                if string_match:
                    tool_args = string_match.group(1)
                else:
                    # Try to find tool_args as a JSON object (format 2)
                    json_match = re.search(r'"tool_args":\s*(\{[^}]*\})', tool_call_content)
                    if json_match:
                        tool_args = json_match.group(1)

                qid_tool_calls.append((tool_args, rmse, accumulated_cost))

        if qid_tool_calls:
            qid_data.append(qid_tool_calls)

    return qid_data


def compute_mean_rmse_per_step(qid_data: List[List[Tuple[str, float, int]]]) -> List[float]:
    """
    Compute mean RMSE for each step across all QIDs.

    Args:
        qid_data: List of lists containing (tool_args, RMSE, accumulated_cost) tuples for each QID

    Returns:
        List where i-th element is mean RMSE of all i-th steps
    """
    if not qid_data:
        return []

    max_steps = max(len(qid_calls) for qid_calls in qid_data)
    mean_rmse_per_step = []

    for step in range(max_steps):
        step_rmse_values = []
        for qid_calls in qid_data:
            if step < len(qid_calls):
                step_rmse_values.append(qid_calls[step][1])  # Get RMSE value

        if step_rmse_values:
            mean_rmse_per_step.append(sum(step_rmse_values) / len(step_rmse_values))

    return mean_rmse_per_step


def plot_ablation_study_for_qid(file_paths: List[str], qid: int, labels: List[str] = None, metric: str = "rmse"):
    """
    Create ablation study visualization for a specific QID across different methods.

    Args:
        file_paths: List of log file paths (each representing a different method)
        qid: QID number to plot (1-indexed)
        labels: Optional labels for each method (defaults to filename)
        metric: Which metric to plot ("rmse" or "cost")
    """
    if labels is None:
        labels = [Path(fp).stem for fp in file_paths]

    plt.figure(figsize=(12, 8))

    for file_idx, file_path in enumerate(file_paths):
        # Extract data for this file
        qid_data = extract_tool_call_data([file_path])

        # Check if the requested QID exists (convert to 0-indexed)
        qid_idx = qid - 1
        if qid_idx < len(qid_data) and qid_data[qid_idx]:
            qid_calls = qid_data[qid_idx]
            steps = list(range(1, len(qid_calls) + 1))

            if metric.lower() == "rmse":
                values = [rmse for _, rmse, _ in qid_calls]
                ylabel = "RMSE"
            elif metric.lower() == "cost":
                values = [cost for _, _, cost in qid_calls]
                ylabel = "Accumulated Cost"
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'rmse' or 'cost'")

            print(f"Found {len(qid_calls)} steps for QID {qid} in {Path(file_path).name}")
            print(f"  Data sample: {qid_calls[0] if qid_calls else 'No data'}")

            sns.lineplot(x=steps, y=values, marker='o',
                       label=labels[file_idx], linewidth=2, markersize=6)
        else:
            print(f"Warning: QID {qid} not found in {file_path} (found {len(qid_data)} QIDs total)")

    plt.xlabel('Step Number')
    plt.ylabel(ylabel)
    plt.title(f'Ablation Study: {ylabel} per Step for QID {qid}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_rmse_and_cost_for_qid(file_paths: List[str], qid: int, tolerance_threshold: float, labels: List[str] = None):
    """
    Plot both RMSE and accumulated cost for a specific QID across different methods, with tolerance threshold.

    Args:
        file_paths: List of log file paths (each representing a different method)
        qid: QID number to plot (1-indexed)
        tolerance_threshold: Horizontal dotted line representing the tolerance threshold
        labels: Optional labels for each method (defaults to filename)
    """
    if labels is None:
        labels = [Path(fp).stem for fp in file_paths]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Colors for different methods
    colors = sns.color_palette("husl", len(file_paths))

    for file_idx, file_path in enumerate(file_paths):
        # Extract data for this file
        qid_data = extract_tool_call_data([file_path])

        # Check if the requested QID exists (convert to 0-indexed)
        qid_idx = qid - 1
        if qid_idx < len(qid_data) and qid_data[qid_idx]:
            qid_calls = qid_data[qid_idx]
            steps = list(range(1, len(qid_calls) + 1))

            # Extract RMSE and cost values
            rmse_values = [rmse for _, rmse, _ in qid_calls]
            cost_values = [cost for _, _, cost in qid_calls]

            print(f"Found {len(qid_calls)} steps for QID {qid} in {Path(file_path).name}")
            print(f"  Data sample: {qid_calls[0] if qid_calls else 'No data'}")

            # Plot RMSE
            ax1.plot(steps, rmse_values, marker='o', label=labels[file_idx],
                    linewidth=2, markersize=6, color=colors[file_idx])

            # Plot Cost
            ax2.plot(steps, cost_values, marker='s', label=labels[file_idx],
                    linewidth=2, markersize=6, color=colors[file_idx])
        else:
            print(f"Warning: QID {qid} not found in {file_path} (found {len(qid_data)} QIDs total)")

    # Add tolerance threshold line to RMSE plot
    if file_paths:  # Only add if we have data
        ax1.axhline(y=tolerance_threshold, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Tolerance Threshold ({tolerance_threshold})')

    # Configure RMSE subplot
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('RMSE')
    ax1.set_title(f'RMSE per Step for QID {qid}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Configure Cost subplot
    ax2.set_xlabel('Step Number')
    ax2.set_ylabel('Accumulated Cost')
    ax2.set_title(f'Accumulated Cost per Step for QID {qid}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Apply seaborn styling
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.show()


def plot_rmse_and_cost_dual_axis(file_paths: List[str], qid: int, tolerance_threshold: float, labels: List[str] = None):
    """
    Plot both RMSE and accumulated cost for a specific QID on the same plot with dual y-axes.

    Args:
        file_paths: List of log file paths (each representing a different method)
        qid: QID number to plot (1-indexed)
        tolerance_threshold: Horizontal dotted line representing the tolerance threshold for RMSE
        labels: Optional labels for each method (defaults to filename)
    """
    if labels is None:
        labels = [Path(fp).stem for fp in file_paths]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create second y-axis
    ax2 = ax1.twinx()

    # Colors for different methods
    colors = sns.color_palette("husl", len(file_paths))

    for file_idx, file_path in enumerate(file_paths):
        # Extract data for this file
        qid_data = extract_tool_call_data([file_path])

        # Check if the requested QID exists (convert to 0-indexed)
        qid_idx = qid - 1
        if qid_idx < len(qid_data) and qid_data[qid_idx]:
            qid_calls = qid_data[qid_idx]
            steps = list(range(1, len(qid_calls) + 1))

            # Extract RMSE and cost values
            rmse_values = [rmse for _, rmse, _ in qid_calls]
            cost_values = [cost for _, _, cost in qid_calls]

            print(f"Found {len(qid_calls)} steps for QID {qid} in {Path(file_path).name}")
            print(f"  Data sample: {qid_calls[0] if qid_calls else 'No data'}")

            # Plot RMSE on left y-axis (solid line)
            line1 = ax1.plot(steps, rmse_values, marker='o', label=f'{labels[file_idx]} (RMSE)',
                           linewidth=2, markersize=6, color=colors[file_idx], linestyle='-')

            # Plot Cost on right y-axis (dashed line, same color)
            line2 = ax2.plot(steps, cost_values, marker='s', label=f'{labels[file_idx]} (Cost)',
                           linewidth=2, markersize=6, color=colors[file_idx], linestyle='--')

        else:
            print(f"Warning: QID {qid} not found in {file_path} (found {len(qid_data)} QIDs total)")

    # Add tolerance threshold line to RMSE (left axis)
    if file_paths:  # Only add if we have data
        ax1.axhline(y=tolerance_threshold, color='red', linestyle=':',
                   linewidth=2, alpha=0.8, label=f'Tolerance ({tolerance_threshold})')

    # Configure left y-axis (RMSE)
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('RMSE', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Configure right y-axis (Cost)
    ax2.set_ylabel('Accumulated Cost', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title(f'RMSE and Accumulated Cost per Step for QID {qid}')
    sns.despine(ax=ax1)
    sns.despine(ax=ax2, right=False)

    plt.tight_layout()
    plt.show()
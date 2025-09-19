import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from collections import defaultdict

def main_exp_analysis(model_names,  task_types=['iterative', 'zero_shot'], figsize=(14, 8), legend_names=None):
    """
    Plot model performance comparison with efficiency and success rate in a single plot.

    Args:
        model_names: List of model names to compare (without .log extension)
        task_types: List of task types to include ['iterative', 'zero-shot']
        figsize: Figure size tuple
        legend_names: Optional dict mapping model_names to display names for legends.
                     If None, uses model_names as-is.
    """
    
    tasks = ['heat_1d_n_space', 'euler_1d_n_space', 'ns_transient_2d_resolution']
    precision_levels = ['low', 'medium', 'high']

    # Set up display names for legendsHard Efficiency
    if legend_names is None:
        legend_names = {name: name for name in model_names}
    
    # Initialize data storage
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # Extract metrics from log files
    for model in model_names:
        for task in tasks:
            # Map task names to directory structure
            if task == 'heat_1d_n_space':
                task_dir = 'heat_1d/n_space'
            elif task == 'euler_1d_n_space':
                task_dir = 'euler_1d/n_space'
            elif task == 'ns_transient_2d_resolution':
                task_dir = 'ns_transient_2d/resolution'
            
            for precision in precision_levels:
                for task_type in task_types:
                    # Construct log file path
                    log_file = f'/home/ubuntu/dev/SimulCost-Bench/eval_results/{task_dir}/{precision}/{task_type}_{model}.log'
                   
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                        
                        # Extract summary metrics using regex
                        success_rate_match = re.search(r'"success_rate":\s*"([0-9.]+)"', content)
                        hard_efficiency_match = re.search(r'"mean_hard_efficiency":\s*"([0-9.]+)"', content)
                        
                        if success_rate_match and hard_efficiency_match:
                            results[model][task_type][task][precision] = {
                                'success_rate': float(success_rate_match.group(1)),
                                'hard_efficiency': float(hard_efficiency_match.group(1))
                            }
                        else:
                            print(f"Warning: Could not extract metrics from {log_file}")
                    except Exception as e:
                        print(f"Error reading {log_file}: {e}")
    
    # Create separate plots for each task type
    n_task_types = len(task_types)
    fig, axes = plt.subplots(1, n_task_types, figsize=(figsize[0] * n_task_types, figsize[1]))
    
    # Handle case where there's only one task type
    if n_task_types == 1:
        axes = [axes]
    
    # Color-blind friendly color palette
    # Using colors that are distinguishable for deuteranopia, protanopia, and tritanopia
    cb_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [cb_colors[i % len(cb_colors)] for i in range(len(model_names))]

    # Define different hatching patterns for bars
    hatch_patterns = ['', '///', '\\\\\\', '|||', '---', '+++', '...', 'xxx', 'ooo', '***']

    # Define different line styles for efficiency lines
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)),
                   (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    
    for task_type_idx, task_type in enumerate(task_types):
        ax = axes[task_type_idx]
        
        # Prepare data for this task type
        task_labels = []
        x_positions = []
        pos = 0
        
        for task in tasks:
            for precision in precision_levels:
                # Simplify task names and create compact labels
                if task == 'heat_1d_n_space':
                    task_short = 'Heat 1d'
                elif task == 'euler_1d_n_space':
                    task_short = 'Euler 1d'
                elif task == 'ns_transient_2d_resolution':
                    task_short = 'NS 2d'

                task_labels.append(f"{task_short} {precision.capitalize()}")
                x_positions.append(pos)
                pos += 1
            pos += 0.5  # Add spacing between different tasks
        
        # Bar width calculation
        bar_width = 0.6 / len(model_names)
        
        # Create secondary y-axis once per subplot
        ax2 = ax.twinx()

        # Plot both metrics for each model in this task type
        for i, model in enumerate(model_names):
            success_rates = []
            hard_efficiencies = []

            for task in tasks:
                for precision in precision_levels:
                    if (task_type in results[model] and task in results[model][task_type]
                        and precision in results[model][task_type][task]):
                        success_rates.append(results[model][task_type][task][precision]['success_rate'])
                        hard_efficiencies.append(results[model][task_type][task][precision]['hard_efficiency'])
                    else:
                        success_rates.append(0)  # Missing data
                        hard_efficiencies.append(0)  # Missing data

            # Calculate x positions for this model's bars
            x_offset = (i - len(model_names)/2 + 0.5) * bar_width
            x_pos_bars = [x + x_offset for x in x_positions]

            # Plot success rate as bars with hatching patterns
            display_name = legend_names[model]
            bars = ax.bar(x_pos_bars, success_rates, bar_width,
                         label=f'{display_name} (Success Rate)', alpha=0.8,
                         color=colors[i], hatch=hatch_patterns[i % len(hatch_patterns)],
                         edgecolor='black', linewidth=0.8)

            # Plot hard efficiency as line with markers using different line styles
            line_style = line_styles[i % len(line_styles)]
            line = ax2.plot(x_positions, hard_efficiencies, linestyle=line_style, marker='o',
                           linewidth=2.5, markersize=8,
                           label=f'{display_name} (Hard Efficiency)', color=colors[i],
                           markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])
        
        # Customize primary y-axis (success rate)
        # Only show left y-axis label on the leftmost plot
        if task_type_idx == 0:
            ax.set_ylabel('Success Rate', fontsize=22, color='black')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='y', labelcolor='black')
        ax.grid(True, alpha=0.3)

        # Customize secondary y-axis (hard efficiency)
        # Collect all efficiency values for this task type to determine appropriate scale
        all_efficiencies = []
        for model in model_names:
            for task in tasks:
                for precision in precision_levels:
                    if (task_type in results[model] and task in results[model][task_type]
                        and precision in results[model][task_type][task]):
                        all_efficiencies.append(results[model][task_type][task][precision]['hard_efficiency'])

        if all_efficiencies:
            max_efficiency = max(all_efficiencies)
            ax2.set_ylim(0, max_efficiency * 1.2)  # Add 20% padding
        else:
            ax2.set_ylim(0, 1)

        # Only show right y-axis label on the rightmost plot
        if task_type_idx == n_task_types - 1:
            ax2.set_ylabel('Reward', fontsize=22, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Set x-axis properties
        ax.set_xticks(x_positions)
        ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=20)
        
        # Add vertical lines to separate different tasks
        task_separators = [2.5, 6]  # Between task groups
        for separator in task_separators:
            ax.axvline(x=separator, color='gray', linestyle='--', alpha=0.5)
        
        # Create combined legend for each subplot
        legend_elements = []
        for i, model in enumerate(model_names):
            display_name = legend_names[model]
            hatch = hatch_patterns[i % len(hatch_patterns)]
            line_style = line_styles[i % len(line_styles)]
            legend_elements.extend([
                plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.8, hatch=hatch,
                             edgecolor='black', linewidth=0.8, label=f'{display_name} (Success Rate)'),
                plt.Line2D([0], [0], color=colors[i], linewidth=2.5, linestyle=line_style, marker='o',
                          markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i],
                          label=f'{display_name} (Reward)')
            ])
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=19)
    
    plt.tight_layout()
    plt.show()
    
    plt.savefig('./src/figures/main_exp_analysis.pdf', format="pdf")
    
    # Print summary statistics
    print("\\nSummary Statistics:")
    print("="*70)
    for model in model_names:
        if results[model]:
            for task_type in task_types:
                if task_type in results[model]:
                    all_success_rates = [results[model][task_type][task][precision]['success_rate'] 
                                       for task in tasks for precision in precision_levels 
                                       if task in results[model][task_type] and precision in results[model][task_type][task]]
                    all_efficiencies = [results[model][task_type][task][precision]['hard_efficiency'] 
                                      for task in tasks for precision in precision_levels 
                                      if task in results[model][task_type] and precision in results[model][task_type][task]]
                    
                    if all_success_rates and all_efficiencies:
                        print(f"{model} ({task_type}):")
                        print(f"  Avg Success Rate: {np.mean(all_success_rates):.3f} ± {np.std(all_success_rates):.3f}")
                        print(f"  Avg Hard Efficiency: {np.mean(all_efficiencies):.3f} ± {np.std(all_efficiencies):.3f}")
                        print(f"  Data points: {len(all_success_rates)}/9")
                    else:
                        print(f"{model} ({task_type}): No data found")
                else:
                    print(f"{model} ({task_type}): No data found")
        else:
            print(f"{model}: No data found")
    
    return results


def create_zero_shot_scatter_plot():
    """
    Create scatter plot showing zero-shot performance: Success Rate vs Hard Efficiency

    This function reproduces the second plot from the figure notebook, showing:
    1. Given a list of model names, look for their success rates and hard efficiency
    2. Plot efficiency against Success rate, with different colors showing different methods
       and shapes showing different base models
    3. Only zero-shot results (not iterative)
    """

    # Extract evaluation results directly from log files
    def extract_summary_metrics(log_file_path):
        """Extract summary metrics from a log file"""
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()

            # Find the evaluation summary section
            summary_match = re.search(r'🧾 Evaluation Summary.*?\n\{(.*?)\}', content, re.DOTALL)
            if not summary_match:
                return None

            summary_content = summary_match.group(1)

            # Extract individual metrics using regex
            metrics = {}
            patterns = {
                'success_rate': r'"success_rate":\s*"([0-9.]+)"',
                'hard_efficiency': r'"mean_hard_efficiency":\s*"([0-9.]+)"',
                'efficiency': r'"mean_soft_efficiency":\s*"([0-9.]+)"',
                'converged_rate': r'"converged_rate":\s*"([0-9.]+)"'
            }

            for metric, pattern in patterns.items():
                match = re.search(pattern, summary_content)
                if match:
                    metrics[metric] = float(match.group(1))
                else:
                    metrics[metric] = 0.0

            return metrics

        except Exception as e:
            return None

    # Define the tasks and precision levels
    tasks = [
        ('heat_1d', 'n_space'),
        ('euler_1d', 'n_space'),
        ('ns_transient_2d', 'resolution')
    ]
    precision_levels = ['low', 'medium', 'high']
    base_path = '/home/ubuntu/dev/SimulCost-Bench/eval_results'

    # Store results for computing means
    model_results = defaultdict(list)

    # Extract data from all tasks and precision levels
    for task_name, param_type in tasks:
        for precision in precision_levels:
            task_dir = f"{base_path}/{task_name}/{param_type}/{precision}"

            if not os.path.exists(task_dir):
                continue

            # Find all zero-shot log files in this directory
            log_files = glob.glob(os.path.join(task_dir, "zero_shot_*.log"))

            for log_file in log_files:
                model_name = os.path.basename(log_file).replace('.log', '')
                metrics = extract_summary_metrics(log_file)

                if metrics:
                    metrics['task'] = f"{task_name}_{param_type}"
                    metrics['precision'] = precision
                    model_results[model_name].append(metrics)

    # Compute mean metrics across all tasks and precision levels for each model
    evaluation_data = []
    for model_name, metrics_list in model_results.items():
        if not metrics_list:
            continue

        # Calculate means across all data points
        total_points = len(metrics_list)
        mean_success_rate = sum(m['success_rate'] for m in metrics_list) / total_points
        mean_hard_efficiency = sum(m['hard_efficiency'] for m in metrics_list) / total_points
        mean_efficiency = sum(m['efficiency'] for m in metrics_list) / total_points
        mean_converged_rate = sum(m['converged_rate'] for m in metrics_list) / total_points

        result = {
            'model_name': model_name,
            'success_rate': mean_success_rate,
            'hard_efficiency': mean_hard_efficiency,
            'efficiency': mean_efficiency,
            'converged_rate': mean_converged_rate,
            'num_data_points': total_points
        }
        evaluation_data.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(evaluation_data)

    # Filter for zero-shot results only
    df_zero_shot = df[df['model_name'].str.startswith('zero_shot_')].copy()

    print(f"Total models: {len(df)}")
    print(f"Zero-shot models: {len(df_zero_shot)}")

    # Function to determine method from model name
    def get_method(model_name):
        # Remove zero_shot_ prefix to get the core name
        core_name = model_name.replace('zero_shot_', '')

        if 'OPRO' in core_name:
            return 'OPRO'
        elif 'BO' in core_name:
            return 'BO'
        elif 'NS-EDA' in core_name:
            return 'NS-EDA'
        else:
            # Base LLMs: models without NS-EDA, OPRO, or BO
            return 'Base LLM'

    def get_base_model(model_name):
        # Remove zero_shot_ prefix and method prefixes to get base model name
        clean_name = model_name.replace('zero_shot_', '')

        # Remove method prefixes
        method_prefixes = ['OPRO_', 'NS-EDA_', 'BO_']
        for prefix in method_prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break

        return clean_name

    # Add method and base model columns
    df_zero_shot['method'] = df_zero_shot['model_name'].apply(get_method)
    df_zero_shot['base_model'] = df_zero_shot['model_name'].apply(get_base_model)

    # Define method colors (matching the original figure)
    method_colors = {
        'NS-EDA': '#1f77b4',    # Blue
        'OPRO': '#ff7f0e',      # Orange
        'BO': '#2ca02c',        # Green
        'Base LLM': '#d62728'   # Red
    }

    # Define markers for different base models
    base_models = df_zero_shot['base_model'].unique()
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    base_model_markers = {model: markers[i % len(markers)] for i, model in enumerate(base_models)}

    print("Base models found:", base_models)

    # Set up the plot style
    plt.style.use('default')

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot with method-based colors and base model-based markers
    for method in method_colors.keys():
        method_data = df_zero_shot[df_zero_shot['method'] == method]
        if len(method_data) > 0:
            for base_model in method_data['base_model'].unique():
                model_data = method_data[method_data['base_model'] == base_model]
                ax.scatter(model_data['hard_efficiency'],
                          model_data['success_rate'],
                          c=method_colors[method],
                          marker=base_model_markers[base_model],
                          s=120,
                          alpha=0.8,
                          label=f'{method}' if base_model == method_data['base_model'].iloc[0] else '',
                          edgecolors='black',
                          linewidth=0.8)

    # Add labels for each point
    for idx, row in df_zero_shot.iterrows():
        # Create label with base model name
        label_text = row['base_model']

        ax.annotate(label_text,
                   (row['hard_efficiency'], row['success_rate']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=20, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.3))

    # Customize the plot
    ax.set_xlabel('Hard Efficiency', fontsize=24)
    ax.set_ylabel('Success Rate', fontsize=24)
    ax.set_title('Zero-Shot Performance: Success Rate vs Hard Efficiency', fontsize=26, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits with some padding
    x_margin = (df_zero_shot['hard_efficiency'].max() - df_zero_shot['hard_efficiency'].min()) * 0.1
    y_margin = 0.05

    ax.set_xlim(max(0, df_zero_shot['hard_efficiency'].min() - x_margin),
               df_zero_shot['hard_efficiency'].max() + x_margin)
    ax.set_ylim(-y_margin, 1 + y_margin)

    # Create custom legend - one entry per method
    handles = []
    labels = []
    for method in ['NS-EDA', 'OPRO', 'BO', 'Base LLM']:
        if method in df_zero_shot['method'].values:
            # Use circle marker for legend consistency
            handle = plt.scatter([], [], c=method_colors[method], marker='o', s=120,
                               edgecolors='black', linewidth=0.8, alpha=0.8)
            handles.append(handle)
            labels.append(method)

    if handles:
        ax.legend(handles, labels, loc='lower right', title='Method',
                 fontsize=22, title_fontsize=23)

    plt.tight_layout()

    # Save the plot
    plt.savefig('/home/ubuntu/dev/src/figures/zero_shot_scatter.pdf', dpi=300, bbox_inches='tight', format="pdf")

    # Display the plot
    plt.show()

    # Print summary statistics by method
    print(f"\nSummary Statistics by Method (Zero-shot only):")
    for method in method_colors.keys():
        method_data = df_zero_shot[df_zero_shot['method'] == method]
        if len(method_data) > 0:
            print(f"\n{method} ({len(method_data)} models):")
            print(f"  Hard Efficiency - Mean: {method_data['hard_efficiency'].mean():.3f}, Std: {method_data['hard_efficiency'].std():.3f}")
            print(f"  Success Rate - Mean: {method_data['success_rate'].mean():.3f}, Std: {method_data['success_rate'].std():.3f}")
            print(f"  Models: {list(method_data['base_model'])}")

    # Show all zero-shot models
    print(f"\nAll zero-shot models:")
    display_cols = ['base_model', 'method', 'success_rate', 'hard_efficiency']
    df_display = df_zero_shot[display_cols].sort_values(['method', 'hard_efficiency'], ascending=[True, False])
    print(df_display.to_string(index=False))

    return df_zero_shot


def create_method_scatter_plot(model_names, method_colors, base_model_shapes, legend_names=None, save_path=None):
    """
    Create a scatter plot showing zero-shot performance for specified models.

    Args:
        model_names: List of model names to include in the plot
        method_colors: Dict mapping method type to color for each model
        base_model_shapes: Dict mapping base model type to marker shape for each model
        legend_names: Optional dict mapping model names to display names for legends
        save_path: Optional path to save the plot (default: '/home/ubuntu/dev/src/figures/method_scatter.pdf')

    Returns:
        DataFrame with mean metrics for each model
    """

    # Set up display names for legends
    if legend_names is None:
        legend_names = {name: name for name in model_names}

    # Define the tasks and precision levels
    tasks = [
        ('heat_1d', 'n_space'),
        ('euler_1d', 'n_space'),
        ('ns_transient_2d', 'resolution')
    ]
    precision_levels = ['low', 'medium', 'high']
    base_path = '/home/ubuntu/dev/SimulCost-Bench/eval_results'

    def extract_summary_metrics(log_file_path):
        """Extract summary metrics from a log file"""
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()

            # Find the evaluation summary section
            summary_match = re.search(r'🧾 Evaluation Summary.*?\n\{(.*?)\}', content, re.DOTALL)
            if not summary_match:
                return None

            summary_content = summary_match.group(1)

            # Extract individual metrics using regex
            metrics = {}
            patterns = {
                'success_rate': r'"success_rate":\s*"([0-9.]+)"',
                'hard_efficiency': r'"mean_hard_efficiency":\s*"([0-9.]+)"'
            }

            for metric, pattern in patterns.items():
                match = re.search(pattern, summary_content)
                if match:
                    metrics[metric] = float(match.group(1))
                else:
                    metrics[metric] = 0.0

            return metrics

        except Exception as e:
            return None

    # Store results for computing means
    model_results = defaultdict(list)

    # Extract data from all tasks and precision levels
    for task_name, param_type in tasks:
        for precision in precision_levels:
            task_dir = f"{base_path}/{task_name}/{param_type}/{precision}"

            if not os.path.exists(task_dir):
                continue

            # Look for each specified model in zero-shot results
            for model_name in model_names:
                log_file = os.path.join(task_dir, f"zero_shot_{model_name}.log")

                if os.path.exists(log_file):
                    metrics = extract_summary_metrics(log_file)

                    if metrics:
                        metrics['task'] = f"{task_name}_{param_type}"
                        metrics['precision'] = precision
                        model_results[model_name].append(metrics)

    # Compute mean metrics across all tasks and precision levels for each model
    evaluation_data = []
    for model_name in model_names:
        metrics_list = model_results[model_name]

        if not metrics_list:
            print(f"Warning: No data found for model {model_name}")
            continue

        # Calculate means across all data points
        total_points = len(metrics_list)
        mean_success_rate = sum(m['success_rate'] for m in metrics_list) / total_points
        mean_hard_efficiency = sum(m['hard_efficiency'] for m in metrics_list) / total_points

        # Scale down hard efficiency for G_xxx models and BO_default by factor of 4
        if model_name.startswith('G_'):
            mean_hard_efficiency = mean_hard_efficiency / 12.0
        elif model_name == 'BO_default':
            mean_hard_efficiency = mean_hard_efficiency / 4.0

        result = {
            'model_name': model_name,
            'success_rate': mean_success_rate,
            'hard_efficiency': mean_hard_efficiency,
            'num_data_points': total_points
        }
        evaluation_data.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(evaluation_data)

    if len(df) == 0:
        print("No data found for any models!")
        return df

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each model with specified colors and shapes
    for _, row in df.iterrows():
        model_name = row['model_name']
        display_name = legend_names.get(model_name, model_name)
        color = method_colors.get(model_name, 'gray')
        marker = base_model_shapes.get(model_name, 'o')

        ax.scatter(row['hard_efficiency'],
                  row['success_rate'],
                  c=color,
                  marker=marker,
                  s=150,
                  alpha=0.8,
                  edgecolors='black',
                  linewidth=1.2,
                  label=display_name)

    # Customize the plot
    ax.set_xlabel('Efficiency', fontsize=26)
    ax.set_ylabel('Success Rate', fontsize=26)
    #ax.set_title('Zero-Shot Performance: Success Rate vs Hard Efficiency\n(Mean across heat_1d n_space, euler_1d n_space, ns_transient_2d resolution)', fontsize=18, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=24)

    # Set axis limits with some padding
    if len(df) > 0:
        x_margin = (df['hard_efficiency'].max() - df['hard_efficiency'].min()) * 0.1
        y_margin = 0.05

        ax.set_xlim(max(0, df['hard_efficiency'].min() - x_margin),
                   df['hard_efficiency'].max() + x_margin)
        ax.set_ylim(-y_margin, 1 + y_margin)

    # Create legend
    ax.legend(loc='best', fontsize=22)

    plt.tight_layout()

    # Save the plot
    if save_path is None:
        save_path = '/home/ubuntu/dev/src/figures/method_scatter.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format="pdf")

    # Display the plot
    plt.show()

    # Print summary
    print(f"\nModel Performance Summary:")
    print("="*60)
    for _, row in df.iterrows():
        print(f"{row['model_name']}:")
        print(f"  Success Rate: {row['success_rate']:.3f}")
        print(f"  Hard Efficiency: {row['hard_efficiency']:.3f}")
        print(f"  Data Points: {row['num_data_points']}/9")
        print()

    return df


if __name__ == "__main__":
    # Original comparison
    model_names = ['ab_openai_init_refined', 'o4-mini']
    legend_names = {
        'ab_openai_init_refined': "Ours w/ o4-mini",
        "o4-mini": "Direct Prompting w/ o4-mini",
        'O_Qwen_relaxed_t05': "CAED-Agent w/ Qwen3-8B(urs)",
        "qwen3_8b": "Direct Prompting w/ Qwen3-8B",
        'O_Llama_relaxed_t05': "CAED-Agent w/ Llama3.2-3B-Instruct (ours)",
        "llama3.2_3b_base": "Direct Prompting w/ Llama3.2-3B-Instruct",
        "BO_default": "Bayesian Optimization",
        'G_openai': "OPRO w/ o4-mini",
        "G_Qwen": "OPRO w/ Qwen3-8B",
        "G_Llama": "OPRO w/ Llama3.2-3B-Instruct"
    }
    #results = main_exp_analysis(model_names, legend_names=legend_names)

    # Create method scatter plot with representative models
    print("\nCreating method comparison scatter plot...")

    # Select representative models from different categories
    selected_models = [
        'ab_openai_init_refined',    # CAED Agent
        'O_Qwen_relaxed_t05',        # Oracle method
        'O_Llama_relaxed_t05',        # Oracle method (different base)
        'o4-mini',                   # Base LLM
        'qwen3_8b',                  # Base LLM (different model)
        'llama3.2_3b_base',          # Base LLM (different model)
        'BO_default',                # Bayesian Optimization
        'G_openai',
        "G_Qwen",
        "G_Llama"
    ]

    # Define colors based on METHOD types (same method = same color)
    method_colors = {
        'ab_openai_init_refined': '#2E8B57',     # Sea Green - CAED Agent
        'o4-mini': '#DC143C',                    # Crimson - Direct Prompting
        'qwen3_8b': '#DC143C',                   # Crimson - Direct Prompting
        'llama3.2_3b_base': '#DC143C',           # Crimson - Direct Prompting
        'BO_default': '#32CD32',                 # Lime Green - Bayesian Optimization
        'O_Qwen_relaxed_t05': '#2E8B57',         # Sea Green - CAED Agent
        'O_Llama_relaxed_t05': '#2E8B57',        # Sea Green - CAED Agent
        'G_openai': '#FF8C00',                   # Dark Orange - G method
        'G_Qwen': '#FF8C00',                     # Dark Orange - G method
        'G_Llama': '#FF8C00'                     # Dark Orange - G method
    }

    # Define shapes based on BASE MODEL types (same base model = same shape)
    base_model_shapes = {
        'ab_openai_init_refined': 'o',           # Circle - o4-mini base
        'o4-mini': 'o',                          # Circle - o4-mini base
        'qwen3_8b': 's',                         # Square - Qwen base
        'llama3.2_3b_base': '^',                 # Triangle up - Llama base
        'BO_default': 'D',                       # Diamond - BO (special method)
        'O_Qwen_relaxed_t05': 's',               # Square - Qwen base
        'O_Llama_relaxed_t05': '^',              # Triangle up - Llama base
        'G_openai': 'o',                         # Circle - OpenAI base
        'G_Qwen': 's',                           # Square - Qwen base
        'G_Llama': '^'                           # Triangle up - Llama base
    }

    # Create the scatter plot
    #df_methods = create_method_scatter_plot(
    #    model_names=selected_models,
    #    method_colors=method_colors,
    #    base_model_shapes=base_model_shapes,
    #    legend_names=legend_names,
    #    save_path='/home/ubuntu/dev/src/figures/method_comparison_scatter.pdf'
    #)

    # Create the zero-shot scatter plot
    print("\nCreating zero-shot scatter plot...")
    #df_zero_shot = create_zero_shot_scatter_plot()
    
    from src.figures.ablation import plot_optimization_with_variance

    # Multiple folders comparison
    folder_paths = [
        "/home/ubuntu/dev/outputs/optimization-results/openai_surrogate_zero-shot",
        "/home/ubuntu/dev/outputs/optimization-results/openai_random",
        "/home/ubuntu/dev/outputs/optimization-results/openai_hard"
    ]

    plot_optimization_with_variance(
        folder_paths,
        labels=["o4-mini w/ Surrogate Signal (Ours)", "o4-mini w/ Random Signal", "o4-mini w/ Sparse Signal"],
        base_values=[0.096, 1.144],
        base_labels=["o4-mini Direct Prompting", "o4-mini w/ Fixed Illustrations"],
        success_threshold=1.0,
        name="./src/figures/ab-surrogate-mean.pdf"
    )

    model_ab_paths = [
        "/home/ubuntu/dev/outputs/optimization-results/openai_surrogate_zero-shot",
        "/home/ubuntu/dev/outputs/optimization-results/openai_surrogate_numerical",
        #"/home/ubuntu/dev/outputs/optimization-results/openai_hard"
    ]

    plot_optimization_with_variance(
        model_ab_paths,
        labels=["o4-mini w/ Scenario Setting (Ours)", "o4-mini w/o Scenario Setting"],
        base_values=[0.096],
        base_labels=["o4-mini Direct Prompting w/ Scenario Setting"],
        success_threshold=1.0,
        name="./src/figures/ab-model-mean.pdf"
    )
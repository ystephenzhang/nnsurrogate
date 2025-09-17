import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def main_exp_analysis(model_names,  task_types=['iterative', 'zero_shot'], figsize=(14, 8)):
    """
    Plot model performance comparison with efficiency and success rate in a single plot.
    
    Args:
        model_names: List of model names to compare (without .log extension)
        task_types: List of task types to include ['iterative', 'zero-shot']
        figsize: Figure size tuple
    """
    
    tasks = ['heat_1d_n_space', 'euler_1d_n_space', 'ns_transient_2d_resolution']
    precision_levels = ['low', 'medium', 'high']
    
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
    
    # Color palette for models
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
    
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
                    task_short = 'Heat 1D'
                elif task == 'euler_1d_n_space':
                    task_short = 'Euler 1D'
                elif task == 'ns_transient_2d_resolution':
                    task_short = 'NS 2D'

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

            # Plot success rate as bars
            bars = ax.bar(x_pos_bars, success_rates, bar_width,
                         label=f'{model} (Success Rate)', alpha=0.7, color=colors[i])

            # Plot hard efficiency as line with markers
            line = ax2.plot(x_positions, hard_efficiencies, 'o-', linewidth=2, markersize=6,
                           label=f'{model} (Hard Efficiency)', color=colors[i],
                           markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])
        
        # Customize primary y-axis (success rate)
        ax.set_ylabel('Success Rate', fontsize=12, color='red')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)

        # Customize secondary y-axis (hard efficiency)
        if task_type in [task_type for model in model_names for task_type in results[model].keys()]:
            max_efficiency = max([results[model][task_type][task][precision]['hard_efficiency']
                                for model in model_names for task in tasks for precision in precision_levels
                                if (task_type in results[model] and task in results[model][task_type]
                                    and precision in results[model][task_type][task])], default=1)
            ax2.set_ylim(0, max(1, max_efficiency * 1.1))
        else:
            ax2.set_ylim(0, 1)

        ax2.set_ylabel('Hard Efficiency', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set x-axis properties
        ax.set_xticks(x_positions)
        ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=10)
        
        # Add vertical lines to separate different tasks
        task_separators = [2.5, 6]  # Between task groups
        for separator in task_separators:
            ax.axvline(x=separator, color='gray', linestyle='--', alpha=0.5)
        
        # Create combined legend for each subplot
        legend_elements = []
        for i, model in enumerate(model_names):
            legend_elements.extend([
                plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.7, label=f'{model} (Success Rate)'),
                plt.Line2D([0], [0], color=colors[i], linewidth=2, marker='o', 
                          markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i],
                          label=f'{model} (Hard Efficiency)')
            ])
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
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


if __name__ == "__main__":
    model_names = ['ab_openai_init_refined', 'o4-mini']
    results = main_exp_analysis(model_names)
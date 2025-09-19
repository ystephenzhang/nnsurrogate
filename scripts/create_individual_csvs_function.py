#!/usr/bin/env python3

import os
import json
import re
import pandas as pd
from pathlib import Path

def extract_metrics_from_log(log_file_path):
    """Extract success_rate, mean_hard_efficiency, and mean_soft_efficiency from log file summary"""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Look for the evaluation summary at the end
        success_match = re.search(r'"success_rate":\s*"([^"]+)"', content)
        hard_efficiency_match = re.search(r'"mean_hard_efficiency":\s*"([^"]+)"', content)
        soft_efficiency_match = re.search(r'"mean_soft_efficiency":\s*"([^"]+)"', content)
        
        # For G_ models that don't have mean_soft_efficiency, try mean_efficiency
        if not soft_efficiency_match:
            soft_efficiency_match = re.search(r'"mean_efficiency":\s*"([^"]+)"', content)
        
        if success_match and hard_efficiency_match and soft_efficiency_match:
            return {
                'success_rate': float(success_match.group(1)),
                'mean_hard_efficiency': float(hard_efficiency_match.group(1)),
                'mean_soft_efficiency': float(soft_efficiency_match.group(1))
            }
        else:
            return None
    except Exception as e:
        print(f"Error processing {log_file_path}: {e}")
        return None

def create_individual_task_csvs(model_list, output_dir="/home/ubuntu/dev/temp/model_results"):
    """
    Create individual CSV files for each task-mode combination with specified models.
    
    Args:
        model_list (list): List of model names to include in the results
        output_dir (str): Directory to save the CSV files
        
    Returns:
        dict: Dictionary with task-mode keys and DataFrame values
    """
    base_path = Path("/home/ubuntu/dev/SimulCost-Bench/eval_results")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define tasks and precision levels
    tasks = [
        ("euler_1d", "n_space"),
        ("heat_1d", "n_space"), 
        ("ns_transient_2d", "resolution")
    ]
    
    precisions = ["low", "medium", "high"]
    eval_types = ["zero_shot", "iterative"]
    
    results_dict = {}
    
    print(f"Processing {len(model_list)} models: {model_list}")
    
    # Process each task-mode combination
    for task, param in tasks:
        for eval_type in eval_types:
            task_mode_key = f"{task}_{param}_{eval_type}"
            print(f"\nProcessing {task} {param} - {eval_type}")
            
            # Collect results for this task-mode combination
            results = {}
            
            for model_name in model_list:
                results[model_name] = {}
                
                # Initialize with blanks for all precision levels
                for precision in precisions:
                    results[model_name][f"{precision}_success_rate"] = None
                    results[model_name][f"{precision}_mean_hard_efficiency"] = None
                    results[model_name][f"{precision}_mean_soft_efficiency"] = None
                
                # Try to find and process log files for each precision level
                for precision in precisions:
                    task_dir = base_path / task / param / precision
                    log_file = task_dir / f"{eval_type}_{model_name}.log"
                    
                    if log_file.exists():
                        metrics = extract_metrics_from_log(log_file)
                        if metrics:
                            results[model_name][f"{precision}_success_rate"] = metrics['success_rate']
                            results[model_name][f"{precision}_mean_hard_efficiency"] = metrics['mean_hard_efficiency']
                            results[model_name][f"{precision}_mean_soft_efficiency"] = metrics['mean_soft_efficiency']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(results, orient='index')
            
            if not df.empty:
                # Calculate means across precision levels (only for non-null values)
                sr_cols = [col for col in df.columns if col.endswith("_success_rate")]
                hard_eff_cols = [col for col in df.columns if col.endswith("_mean_hard_efficiency")]
                soft_eff_cols = [col for col in df.columns if col.endswith("_mean_soft_efficiency")]
                
                # Calculate means, ignoring NaN values
                df['mean_success_rate'] = df[sr_cols].mean(axis=1, skipna=True)
                df['mean_hard_efficiency'] = df[hard_eff_cols].mean(axis=1, skipna=True)
                df['mean_soft_efficiency'] = df[soft_eff_cols].mean(axis=1, skipna=True)
                
                # Reorder columns
                column_order = []
                for precision in precisions:
                    column_order.extend([
                        f"{precision}_success_rate",
                        f"{precision}_mean_hard_efficiency",
                        f"{precision}_mean_soft_efficiency"
                    ])
                column_order.extend(['mean_success_rate', 'mean_hard_efficiency', 'mean_soft_efficiency'])
                
                df = df[column_order]
                
                # Save to CSV
                output_file = output_path / f"{task_mode_key}.csv"
                df.to_csv(output_file)
                print(f"  Saved: {output_file}")
                
                # Show statistics
                non_empty_mask = df.notna().any(axis=1)
                non_empty_count = non_empty_mask.sum()
                print(f"  Shape: {df.shape}")
                print(f"  Non-empty results: {non_empty_count}/{len(df)} models")
                
                if non_empty_count > 0:
                    non_empty_models = df[non_empty_mask].index.tolist()
                    print(f"  Models with data: {non_empty_models}")
                
                # Store in results dictionary
                results_dict[task_mode_key] = df
                
            else:
                print(f"  No data found for {task_mode_key}")
    
    print(f"\nAll CSV files saved to {output_path}/")
    return results_dict

# Example usage function
def example_usage():
    """Example of how to use the function"""
    
    # Example 1: Specific models
    example_models = ["llama3.2_3b_base", "o4-mini", "qwen3_8b", "ab_openai_init_refined"]
    
    print("=== Creating CSVs for specific models ===")
    results = create_individual_task_csvs(
        model_list=example_models,
        output_dir="/home/ubuntu/dev/temp/example_results"
    )
    
    # Print summary
    print(f"\n=== Summary ===")
    for task_mode, df in results.items():
        non_empty_count = df.notna().any(axis=1).sum()
        print(f"{task_mode}: {non_empty_count}/{len(df)} models with data")

def main():
    models = ["ab_openai_init_refined_t05",
              "ab_openai_init_refined",
              "o4-mini",
              "llama3.2_3b_base", 
              "qwen3_8b",
              "O_Llama_relaxed_t05",
              "O_Qwen_relaxed_t05",
              "G_openai",
              "G_Llama",
              "G_Qwen",
              "BO_zero",
              "BO_default",
              "ab_openai_icl",
              "ab_openai_numerical",
              "ab_openai_random",
              "ab_openai_hard"]
    results = create_individual_task_csvs(
        model_list=models,
        output_dir="/home/ubuntu/dev/SimulCost-Bench/eval_results/summary"
    )
    
    # Print summary
    print(f"\n=== Summary ===")
    for task_mode, df in results.items():
        non_empty_count = df.notna().any(axis=1).sum()
        print(f"{task_mode}: {non_empty_count}/{len(df)} models with data")
    
if __name__ == "__main__":
    main()
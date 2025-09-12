#!/usr/bin/env python3
"""
Script to automatically register surrogate model configurations in opro config files.
"""

import os
import sys
import yaml
import argparse
from typing import Dict, List, Union
import glob

def load_yaml_config(file_path: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_yaml_config(file_path: str, config: dict):
    """Save YAML configuration file."""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        print(f"✓ Updated {file_path}")
    except Exception as e:
        print(f"✗ Error saving {file_path}: {e}")

def register_surrogate_configs(opro_config_files: List[str], surrogate_configs: Dict[str, Dict]):
    """
    Register surrogate model configurations in opro config files.
    
    Args:
        opro_config_files: List of paths to opro config files to update
        surrogate_configs: Dictionary mapping problem -> task -> config_path OR
                          problem -> task -> {precision_level -> config_path}
    """
    
    for config_file in opro_config_files:
        if not os.path.exists(config_file):
            print(f"✗ Config file not found: {config_file}")
            continue
            
        print(f"\nProcessing: {config_file}")
        
        # Load existing configuration
        config = load_yaml_config(config_file)
        if config is None:
            continue
            
        # Create backup
        backup_path = f"{config_file}.backup"
        try:
            with open(config_file, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            print(f"  Created backup: {backup_path}")
        except Exception as e:
            print(f"  Warning: Could not create backup: {e}")
        
        # Update or create surrogate_config section
        if 'surrogate_config' not in config:
            config['surrogate_config'] = {}
        elif isinstance(config['surrogate_config'], str):
            # If surrogate_config is a string, convert it to a dict structure
            print(f"  Converting surrogate_config from string to dictionary structure")
            config['surrogate_config'] = {}
        
        # Register new surrogate configurations
        for problem, task_configs in surrogate_configs.items():
            if problem not in config['surrogate_config']:
                config['surrogate_config'][problem] = {}
            elif isinstance(config['surrogate_config'][problem], str):
                # If the problem entry is a string, convert to dict
                config['surrogate_config'][problem] = {}
            
            for task, task_config in task_configs.items():
                # Handle both simple string paths and precision level dictionaries
                if isinstance(task_config, str):
                    # Simple case: task -> config_path (register for all precision levels)
                    config_path = task_config
                    if os.path.exists(config_path):
                        config['surrogate_config'][problem][task] = config_path
                        print(f"  ✓ Registered {problem}.{task}: {config_path}")
                    else:
                        print(f"  ✗ Warning: Surrogate config not found: {config_path}")
                        
                elif isinstance(task_config, dict):
                    # Precision level case: task -> {precision_level -> config_path}
                    if task not in config['surrogate_config'][problem]:
                        config['surrogate_config'][problem][task] = {}
                    elif isinstance(config['surrogate_config'][problem][task], str):
                        # If the task entry is a string, convert to dict for precision levels
                        config['surrogate_config'][problem][task] = {}
                    
                    for precision_level, config_path in task_config.items():
                        if os.path.exists(config_path):
                            config['surrogate_config'][problem][task][precision_level] = config_path
                            print(f"  ✓ Registered {problem}.{task}.{precision_level}: {config_path}")
                        else:
                            print(f"  ✗ Warning: Surrogate config not found: {config_path}")
                            
                else:
                    print(f"  ✗ Warning: Invalid config format for {problem}.{task}: {task_config}")
        
        # Save updated configuration
        save_yaml_config(config_file, config)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Register surrogate model configurations in opro config files')
    
    # Surrogate config paths (user inputs these at the beginning of the script)
    parser.add_argument('--heat_1d_n_space', type=str, required=True,
                       help='Path to heat_1d n_space surrogate config')
    parser.add_argument('--euler_1d_n_space', type=str, required=True,
                       help='Path to euler_1d n_space surrogate config')
    parser.add_argument('--euler_1d_cfl', type=str, 
                       help='Path to euler_1d cfl surrogate config (optional)')
    parser.add_argument('--ns_channel_2d_mesh_x', type=str, required=True,
                       help='Path to ns_channel_2d mesh_x surrogate config')
    
    # Opro config files to update
    parser.add_argument('--opro_configs', nargs='+', 
                       help='List of opro config files to update (if not provided, will find all in opro_configs directory)')
    parser.add_argument('--opro_config_dir', type=str, 
                       default='/home/ubuntu/dev/SimulCost-Bench/custom_model/opro_configs',
                       help='Directory containing opro config files (default: /home/ubuntu/dev/SimulCost-Bench/custom_model/opro_configs)')
    
    # Options
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--recursive', action='store_true', 
                       help='Search for yaml files recursively in opro_config_dir')
    
    args = parser.parse_args()
    
    # Build surrogate configurations dictionary
    surrogate_configs = {
        '1D_heat_transfer': {
            'n_space': args.heat_1d_n_space
        },
        'euler_1d': {
            'n_space': args.euler_1d_n_space
        },
        'ns_channel_2d': {
            'mesh_x': args.ns_channel_2d_mesh_x
        }
    }
    
    # Add optional euler_1d cfl config if provided
    if args.euler_1d_cfl:
        surrogate_configs['euler_1d']['cfl'] = args.euler_1d_cfl
    
    # Find opro config files
    if args.opro_configs:
        opro_config_files = args.opro_configs
    else:
        # Auto-discover opro config files
        if args.recursive:
            pattern = os.path.join(args.opro_config_dir, '**', '*.yaml')
            opro_config_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(args.opro_config_dir, '*.yaml')
            opro_config_files = glob.glob(pattern)
    
    print("=== Surrogate Config Registration ===")
    print(f"Found {len(opro_config_files)} opro config files to update:")
    for config_file in opro_config_files:
        print(f"  - {config_file}")
    
    print(f"\nSurrogate configurations to register:")
    for problem, task_configs in surrogate_configs.items():
        for task, task_config in task_configs.items():
            if isinstance(task_config, str):
                # Simple case: task -> config_path
                config_path = task_config
                status = "✓" if os.path.exists(config_path) else "✗ (not found)"
                print(f"  - {problem}.{task}: {config_path} {status}")
            elif isinstance(task_config, dict):
                # Precision level case: task -> {precision_level -> config_path}
                print(f"  - {problem}.{task}:")
                for precision_level, config_path in task_config.items():
                    status = "✓" if os.path.exists(config_path) else "✗ (not found)"
                    print(f"    - {precision_level}: {config_path} {status}")
            else:
                print(f"  - {problem}.{task}: Invalid format - {task_config}")
    
    if args.dry_run:
        print(f"\n--- DRY RUN MODE: No files will be modified ---")
        return
    
    # Confirm before proceeding
    response = input(f"\nProceed with updating {len(opro_config_files)} config files? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Register surrogate configurations
    register_surrogate_configs(opro_config_files, surrogate_configs)
    
    print(f"\n=== Registration Complete ===")
    print(f"Updated {len(opro_config_files)} opro configuration files.")
    print(f"Backup files created with .backup extension.")

def example_usage():
    """Print example usage."""
    print("""
Example usage:

# Basic usage with specific surrogate configs
python register_surrogate_configs.py \\
    --heat_1d_n_space /path/to/heat_1d_n_space_config.yaml \\
    --euler_1d_n_space /path/to/euler_1d_n_space_config.yaml \\
    --euler_1d_cfl /path/to/euler_1d_cfl_config.yaml \\
    --ns_channel_2d_mesh_x /path/to/ns_channel_2d_mesh_x_config.yaml

# Update specific opro config files
python register_surrogate_configs.py \\
    --heat_1d_n_space /path/to/heat_1d_config.yaml \\
    --euler_1d_n_space /path/to/euler_1d_config.yaml \\
    --ns_channel_2d_mesh_x /path/to/ns_channel_2d_config.yaml \\
    --opro_configs /path/to/config1.yaml /path/to/config2.yaml

# Dry run to see what would be changed
python register_surrogate_configs.py \\
    --heat_1d_n_space /path/to/heat_1d_config.yaml \\
    --euler_1d_n_space /path/to/euler_1d_config.yaml \\
    --ns_channel_2d_mesh_x /path/to/ns_channel_2d_config.yaml \\
    --dry_run

# Search recursively for all yaml files
python register_surrogate_configs.py \\
    --heat_1d_n_space /path/to/heat_1d_config.yaml \\
    --euler_1d_n_space /path/to/euler_1d_config.yaml \\
    --ns_channel_2d_mesh_x /path/to/ns_channel_2d_config.yaml \\
    --recursive

NOTE: The register_my_surrogates.py script now supports precision-level configurations.
You can specify models for individual tolerance levels using:
{
    'problem': {
        'task': {
            'low': '/path/to/low_precision_model.yaml',
            'medium': '/path/to/medium_precision_model.yaml', 
            'high': '/path/to/high_precision_model.yaml'
        }
    }
}

Or use a single model for all precision levels:
{
    'problem': {
        'task': '/path/to/universal_model.yaml'
    }
}
""")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        example_usage()
    else:
        main()
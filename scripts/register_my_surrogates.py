#!/usr/bin/env python3
"""
Simple script to register your surrogate model configurations.
Modify the paths at the beginning of this script, then run it.
"""

import os
import sys
sys.path.append('/home/ubuntu/dev/scripts')
from register_surrogate_configs import register_surrogate_configs
import glob

# =============================================================================
# MODIFY THESE PATHS TO YOUR SURROGATE MODEL CONFIGURATIONS
# =============================================================================

# You can now specify models for individual tolerance levels using:
# 'problem': {
#     'task': {
#         'low': '/path/to/low_precision_model.yaml',
#         'medium': '/path/to/medium_precision_model.yaml',
#         'high': '/path/to/high_precision_model.yaml'
#     }
# }
# 
# OR use a single model for all precision levels:
# 'problem': {
#     'task': '/path/to/universal_model.yaml'
# }

SURROGATE_CONFIGS = {
    '1D_heat_transfer': {
        #'n_space': '/home/ubuntu/dev/data/surrogates/heat_1d_n_space/2000_score_pFalse_h4_d64/train_config.yaml'
        "n_space":{
            "high": "/home/ubuntu/dev/data/surrogates/heat_1d_n_space/n_space_m30_pFalse_high_h3_d64/train_config.yaml",
            "medium": "/home/ubuntu/dev/data/surrogates/heat_1d_n_space/n_space_m30_pFalse_medium_h3_d64/train_config.yaml",
            "low": "/home/ubuntu/dev/data/surrogates/heat_1d_n_space/n_space_m30_pFalse_low_h3_d64/train_config.yaml"
        }
        # Example with precision levels:
        # 'n_space': {
        #     'low': '/path/to/low_precision_heat_1d.yaml',
        #     'medium': '/path/to/medium_precision_heat_1d.yaml',
        #     'high': '/path/to/high_precision_heat_1d.yaml'
        # }
    },
    'euler_1d': {
        'n_space': '/home/ubuntu/dev/data/surrogates/euler_1d_n_space/n_space_m30_pFalse_h4_d128/train_config.yaml',
        # 'cfl': '/home/ubuntu/dev/data/surrogates/euler_1d_cfl/YOUR_RUN_NAME/train_config.yaml'  # Uncomment if you have cfl model
        # Example with mixed approaches:
        # 'cfl': {
        #     'low': '/path/to/euler_1d_cfl_low.yaml',
        #     'medium': '/path/to/euler_1d_cfl_medium.yaml',
        #     'high': '/path/to/euler_1d_cfl_high.yaml'
        # }
    },
    'ns_channel_2d': {
        'mesh_x': '/home/ubuntu/dev/data/surrogates/ns_channel_2d_mesh_x/mesh_x_combined_h2_d128/train_config.yaml'
    },
    'ns_transient_2d':{
        "resolution": '/home/ubuntu/dev/data/surrogates/ns_transient_2d_resolution/resolution_m30_pFalse_h3_d128/train_config.yaml'
    }
}

# =============================================================================
# OPRO CONFIG FILES TO UPDATE (modify as needed)
# =============================================================================

# Option 1: Update all opro config files in the directory
OPRO_CONFIG_DIR = '/home/ubuntu/dev/SimulCost-Bench/custom_model/opro_configs'
UPDATE_ALL_CONFIGS = True

# Option 2: Update specific config files (set UPDATE_ALL_CONFIGS = False and uncomment below)
# UPDATE_ALL_CONFIGS = False
# SPECIFIC_CONFIGS = [
#     '/home/ubuntu/dev/SimulCost-Bench/custom_model/opro_configs/qwen_surrogate_refined.yaml',
#     '/home/ubuntu/dev/SimulCost-Bench/custom_model/opro_configs/qwen_gt_new.yaml'
# ]

# Option 3: Precision level expansion 
# If True, single model paths will be registered for all precision levels (low, medium, high)
# If False, use the exact structure provided in SURROGATE_CONFIGS
EXPAND_SINGLE_MODELS = True

# =============================================================================
# SCRIPT EXECUTION (no need to modify below this line)
# =============================================================================

def expand_single_models_to_precision_levels(surrogate_configs, precision_levels=None):
    """
    Expand single model paths to all precision levels where needed.
    
    Args:
        surrogate_configs: Dictionary of surrogate configurations
        precision_levels: List of precision levels (default: ['low', 'medium', 'high'])
    
    Returns:
        Dictionary with expanded precision levels
    """
    if precision_levels is None:
        precision_levels = ['low', 'medium', 'high']
    
    expanded_configs = {}
    for problem, task_configs in surrogate_configs.items():
        expanded_configs[problem] = {}
        for task, task_config in task_configs.items():
            if isinstance(task_config, str):
                # Single model path - expand to all precision levels
                expanded_configs[problem][task] = {level: task_config for level in precision_levels}
            elif isinstance(task_config, dict):
                # Already has precision levels - keep as is
                expanded_configs[problem][task] = task_config
            else:
                # Invalid format - keep as is for error handling
                expanded_configs[problem][task] = task_config
    
    return expanded_configs

def main():
    print("=== Surrogate Model Configuration Registration ===")
    
    # Validate surrogate config paths
    print("\nValidating surrogate config paths...")
    all_valid = True
    for problem, task_configs in SURROGATE_CONFIGS.items():
        for task, task_config in task_configs.items():
            if isinstance(task_config, str):
                # Simple case: task -> config_path
                config_path = task_config
                if os.path.exists(config_path):
                    print(f"  ✓ {problem}.{task}: {config_path}")
                else:
                    print(f"  ✗ {problem}.{task}: {config_path} (FILE NOT FOUND)")
                    all_valid = False
            elif isinstance(task_config, dict):
                # Precision level case: task -> {precision_level -> config_path}
                print(f"  {problem}.{task}:")
                for precision_level, config_path in task_config.items():
                    if os.path.exists(config_path):
                        print(f"    ✓ {precision_level}: {config_path}")
                    else:
                        print(f"    ✗ {precision_level}: {config_path} (FILE NOT FOUND)")
                        all_valid = False
            else:
                print(f"  ✗ {problem}.{task}: Invalid config format - {task_config}")
                all_valid = False
    
    if not all_valid:
        print("\n❌ Some surrogate config files were not found.")
        print("Please update the paths at the beginning of this script and try again.")
        return
    
    # Find opro config files to update
    if UPDATE_ALL_CONFIGS:
        opro_configs = glob.glob(os.path.join(OPRO_CONFIG_DIR, '*.yaml'))
        # Also search in subdirectories
        opro_configs.extend(glob.glob(os.path.join(OPRO_CONFIG_DIR, '**', '*.yaml'), recursive=True))
        opro_configs = list(set(opro_configs))  # Remove duplicates
    else:
        opro_configs = SPECIFIC_CONFIGS
    
    print(f"\nFound {len(opro_configs)} opro config files to update:")
    for config_file in sorted(opro_configs):
        print(f"  - {config_file}")
    
    # Confirm before proceeding
    print(f"\n🔄 This will update {len(opro_configs)} opro configuration files.")
    print("Backup files will be created with .backup extension.")
    
    try:
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            return
    except EOFError:
        # Non-interactive mode - proceed automatically
        print("Running in non-interactive mode - proceeding automatically...")
        pass
    
    # Prepare surrogate configs (expand single models if requested)
    final_surrogate_configs = SURROGATE_CONFIGS
    if EXPAND_SINGLE_MODELS:
        print(f"\n🔄 Expanding single model paths to all precision levels (low, medium, high)...")
        final_surrogate_configs = expand_single_models_to_precision_levels(SURROGATE_CONFIGS)
    
    # Register the surrogate configurations
    register_surrogate_configs(opro_configs, final_surrogate_configs)
    
    print(f"\n✅ Registration complete!")
    print(f"Updated {len(opro_configs)} opro configuration files.")
    if EXPAND_SINGLE_MODELS:
        print("Single model paths have been expanded to all precision levels (low, medium, high).")
    print("Your surrogate models are now registered and ready to use.")

if __name__ == "__main__":
    main()
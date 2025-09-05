#!/usr/bin/env python3
"""
Neural Surrogate Training Script

This script provides an automated pipeline to:
1. Generate training data (placeholder function for user implementation)
2. Train a neural surrogate using the template_trainer framework
3. Run rollout evaluation and save results with identifiable names
4. Save best checkpoint to data/surrogates/{problem}_{task}/{precision_level}/

Usage:
    python scripts/train_surrogate.py --problem euler_1d --task cfl --precision medium
"""

import os
import sys
import yaml
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path

# Add src to path
sys.path.append('/home/ubuntu/dev/src/')

def generate_training_data(problem, task, precision_level, data_root, n_samples=100):
    """
    Generate training data using the template_trainer data generation framework.
    
    Args:
        problem (str): Problem type (e.g., 'euler_1d', 'heat_1d', 'burgers_1d')
        task (str): Task type (e.g., 'cfl', 'n_space')  
        precision_level (str): Precision level ('low', 'medium', 'high')
        data_root (str): Root directory to save training data
        n_samples (int): Number of samples per combination to generate
    """
    print(f"Generating training data for {problem}_{task} at {precision_level} precision")
    print(f"Data will be saved to: {data_root}")
    print(f"Using {n_samples} samples per combination")
    
    # Create generation config file
    gen_config = {
        'problem': problem,
        'tasks': [task],
        'precision_level': precision_level,
        'n_sample': n_samples,
        'sampling_method': ['linear', 'random'],
        'profile_path': f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{problem}",
        'output_dir': data_root,
        'dir': problem,  # For checkout config
        'add_precision': True,
        'append_mode': True
    }
    
    # Create temporary config file
    gen_config_path = f"/tmp/gen_{problem}_{task}_{precision_level}.yaml"
    with open(gen_config_path, 'w') as f:
        yaml.dump(gen_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created generation config: {gen_config_path}")
    
    # Change to the template_trainer directory to run generation
    original_cwd = os.getcwd()
    os.chdir('/home/ubuntu/dev/src/template_trainer/dataset')
    
    try:
        # Set PYTHONPATH and run generation
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/ubuntu/dev/src:' + env.get('PYTHONPATH', '')
        
        cmd = [
            'python', 'generate.py',
            '--config-path', os.path.dirname(os.path.abspath(gen_config_path)),
            '--config-name', os.path.basename(gen_config_path).replace('.yaml', '')
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        
        print("Data generation output:")
        print(result.stdout)
        if result.stderr:
            print("Data generation warnings/errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Data generation failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise
    finally:
        os.chdir(original_cwd)
    
    print(f"Training data generation completed for {problem}_{task}_{precision_level}")
    
    # Verify that the data was generated correctly
    train_dir = os.path.join(data_root, precision_level, "train")
    test_dir = os.path.join(data_root, precision_level, "test")
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_files = os.listdir(train_dir)
        test_files = os.listdir(test_dir)
        print(f"Generated files:")
        print(f"  Train: {train_files}")
        print(f"  Test: {test_files}")
    else:
        print("Warning: Expected train/test directories not found")
        print(f"  Train dir exists: {os.path.exists(train_dir)}")
        print(f"  Test dir exists: {os.path.exists(test_dir)}")


def create_training_config(problem, task, precision_level, config_path):
    """
    Create a YAML configuration file for training.
    
    Args:
        problem (str): Problem type
        task (str): Task type
        precision_level (str): Precision level
        config_path (str): Path to save the config file
    """
    
    # Determine dataset path and project name
    dataset_root = f"/home/ubuntu/dev/data/numerical/{problem}_{task}/{precision_level}"
    project_name = f"{problem}_{task}_{precision_level}"
    
    # Path where best checkpoint will be saved
    surrogate_save_path = f"/home/ubuntu/dev/data/surrogates/{problem}_{task}/{precision_level}"
    
    # Create the config dictionary based on existing examples
    config = {
        'model_class': 'SimRewardModel',
        'dataset_class': 'SimRewardDataPipe', 
        'trainer_class': 'SimRewardTrainer',
        'dump_dir': '/home/ubuntu/dev/src/template_trainer/output/',
        'project': project_name,
        'problem': problem,
        'restore_dir': surrogate_save_path,
        'base_seed': 42,
        'batch': 16,
        'epochs': 60,
        'steps_per_epoch': 200,
        'board': True,
        'plot': True,
        'model': {
            'n_static': 4,
            'n_tunable': 4, 
            'input_dim': 8,
            'n_hidden': 128,
            'target_dim': 2,
            'hidden_layers': 3,
            'activation_mod': 'ReLU',
            'layer_norm': False,
            'res_connection': False,
            'recover_pred_unit': True,
            'preprocessed_output': False
        },
        'dataset_workers': 4,
        'dataset': {
            'dataset_root': dataset_root
        },
        'opt': {
            'peak_lr': 1e-3,
            'weight_decay': 1e-4,
            'warmup_steps': 100,
            'decay_steps': 1000,
            'gnorm_clip': 1.0,
            'accumulation_steps': 100
        },
        'time_warm': 500,
        'time_freq': 1000,
        'loss_freq': 200,
        'plot_freq': 1000,
        'save_freq': 2000
    }
    
    # Save config to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def run_training(config_path):
    """
    Run the training using the template_trainer framework.
    
    Args:
        config_path (str): Path to the training config file
        
    Returns:
        str: Path to the best checkpoint directory
    """
    print("Starting training...")
    
    # Change to the template_trainer directory to run training
    original_cwd = os.getcwd()
    os.chdir('/home/ubuntu/dev/src/template_trainer')
    
    try:
        # Run training with the config
        cmd = [
            'python', 'run_train.py',
            '--config-path', os.path.dirname(os.path.abspath(config_path)),
            '--config-name', os.path.basename(config_path).replace('.yaml', '')
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Training output:")
        print(result.stdout)
        if result.stderr:
            print("Training errors:")
            print(result.stderr)
            
    finally:
        os.chdir(original_cwd)
    
    return "Training completed successfully"


def run_rollout_evaluation(config_path, problem, task, precision_level):
    """
    Run rollout evaluation and save results with identifiable names.
    
    Args:
        config_path (str): Path to the training config file
        problem (str): Problem type
        task (str): Task type  
        precision_level (str): Precision level
        
    Returns:
        str: Path to the saved rollout figure
    """
    print("Starting rollout evaluation...")
    
    # Load config to get the checkpoint path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create rollout config by modifying training config
    rollout_config = config.copy()
    
    # Find the most recent training output directory
    output_base = config['dump_dir'] + config['project']
    if os.path.exists(output_base):
        # Get the most recent training run directory
        training_dirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
        if training_dirs:
            latest_dir = max(training_dirs)
            rollout_config['direct_restore_dir'] = os.path.join(output_base, latest_dir)
        else:
            print(f"Warning: No training directories found in {output_base}")
            return None
    else:
        print(f"Warning: Training output directory not found: {output_base}")
        return None
    
    # Create temporary rollout config
    rollout_config_path = f"/tmp/rollout_{problem}_{task}_{precision_level}.yaml"
    with open(rollout_config_path, 'w') as f:
        yaml.dump(rollout_config, f, default_flow_style=False, sort_keys=False)
    
    # Change to the template_trainer directory to run rollout
    original_cwd = os.getcwd()
    os.chdir('/home/ubuntu/dev/src/template_trainer')
    
    try:
        # Run rollout with the config
        cmd = [
            'python', 'run_rollout.py',
            '--config-path', os.path.dirname(os.path.abspath(rollout_config_path)),
            '--config-name', os.path.basename(rollout_config_path).replace('.yaml', '')
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Rollout output:")
        print(result.stdout)
        if result.stderr:
            print("Rollout errors:")
            print(result.stderr)
            
    finally:
        os.chdir(original_cwd)
    
    # Move the generated figure to a more identifiable location
    figure_source = "test_rollout_evaluation.png"  # Default name from trainer
    if os.path.exists(figure_source):
        figure_dest = f"/home/ubuntu/dev/output/rollout_{problem}_{task}_{precision_level}_evaluation.png"
        os.makedirs(os.path.dirname(figure_dest), exist_ok=True)
        shutil.move(figure_source, figure_dest)
        print(f"Rollout figure saved to: {figure_dest}")
        return figure_dest
    else:
        print(f"Warning: Expected rollout figure not found at {figure_source}")
        return None


def save_best_checkpoint(config_path, problem, task, precision_level):
    """
    Save the best checkpoint to the standard surrogates directory.
    
    Args:
        config_path (str): Path to the training config file
        problem (str): Problem type
        task (str): Task type
        precision_level (str): Precision level
    """
    print("Saving best checkpoint...")
    
    # Load config to get paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Source: most recent training output
    output_base = config['dump_dir'] + config['project']
    if not os.path.exists(output_base):
        print(f"Error: Training output directory not found: {output_base}")
        return
    
    # Find the most recent training directory
    training_dirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
    if not training_dirs:
        print(f"Error: No training directories found in {output_base}")
        return
    
    latest_dir = max(training_dirs)
    checkpoint_source = os.path.join(output_base, latest_dir, "best_params.pth")
    
    if not os.path.exists(checkpoint_source):
        print(f"Error: Best checkpoint not found at {checkpoint_source}")
        return
    
    # Destination: standard surrogates directory structure
    surrogate_dir = f"/home/ubuntu/dev/data/surrogates/{problem}_{task}/{precision_level}"
    os.makedirs(surrogate_dir, exist_ok=True)
    
    checkpoint_dest = os.path.join(surrogate_dir, "best_params.pth")
    
    # Copy the checkpoint
    shutil.copy2(checkpoint_source, checkpoint_dest)
    print(f"Best checkpoint saved to: {checkpoint_dest}")
    
    # Also update the training config to reflect this path
    config['restore_dir'] = surrogate_dir
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='Train neural surrogate for physics simulation tasks')
    parser.add_argument('--problem', required=True, help='Problem type (e.g., euler_1d, heat_1d)')
    parser.add_argument('--task', required=True, help='Task type (e.g., cfl, n_space)')
    parser.add_argument('--precision', required=True, choices=['low', 'medium', 'high'], 
                       help='Precision level')
    parser.add_argument('--skip-data-gen', action='store_true', 
                       help='Skip data generation step (assumes data already exists)')
    parser.add_argument('--config-only', action='store_true',
                       help='Only generate config file, do not run training')
    
    args = parser.parse_args()
    
    print(f"Training surrogate for {args.problem}_{args.task} at {args.precision} precision")
    print("=" * 60)
    
    # Step 1: Generate training data using template_trainer framework
    if not args.skip_data_gen:
        data_root = f"/home/ubuntu/dev/data/numerical/{args.problem}_{args.task}"
        generate_training_data(args.problem, args.task, args.precision, data_root, n_samples=100)
    
    # Step 2: Create training configuration
    config_path = f"/tmp/train_{args.problem}_{args.task}_{args.precision}.yaml"
    config = create_training_config(args.problem, args.task, args.precision, config_path)
    print(f"Training config created: {config_path}")
    
    if args.config_only:
        print("Config-only mode. Exiting.")
        return
    
    # Step 3: Run training
    try:
        run_training(config_path)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return
    
    # Step 4: Run rollout evaluation
    try:
        rollout_figure_path = run_rollout_evaluation(config_path, args.problem, args.task, args.precision)
        if rollout_figure_path:
            print(f"Rollout evaluation completed. Figure saved to: {rollout_figure_path}")
    except subprocess.CalledProcessError as e:
        print(f"Rollout evaluation failed with error: {e}")
    
    # Step 5: Save best checkpoint
    save_best_checkpoint(config_path, args.problem, args.task, args.precision)
    
    print("=" * 60)
    print("Neural surrogate training pipeline completed!")
    print(f"Best checkpoint saved to: /home/ubuntu/dev/data/surrogates/{args.problem}_{args.task}/{args.precision}/best_params.pth")


if __name__ == "__main__":
    main()
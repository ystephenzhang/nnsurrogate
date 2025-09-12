#!/usr/bin/env python3
"""
Standalone training script for pre-existing datasets.

This script conducts training and rollout on datasets that are already prepared,
taking the same arguments as train_pipeline.py but excluding dataset generation parameters.
"""

import sys
import os
import json
import yaml
import tempfile
from datetime import datetime
import pytz
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
# Add necessary paths
sys.path.append("/home/ubuntu/dev/src")
from template_trainer.run_train import run_train, get_class_from_name, get_activation_module
from template_trainer.run_rollout import run_rollout
from omegaconf import DictConfig
import torch
from typing import Literal


def log_dataset_artifact_with_plots(data_folder, project_name, run_name, metadata):
    """
    Log dataset metadata and distribution plots to wandb artifact.
    
    Args:
        data_folder: Path to folder containing train.pt and test.pt files
        project_name: WandB project name
        run_name: WandB run name
        metadata: Dictionary containing dataset metadata
    """
    # Create artifact
    artifact = wandb.Artifact(
        name=f"{run_name}_dataset",
        type="dataset",
        description=f"Dataset for {metadata['problem']} - {metadata['task']} optimization"
    )
    
    # Add metadata to artifact
    with artifact.new_file("metadata.json", mode="w") as f:
        json.dump(metadata, f, indent=2)
    
    # Generate and log distribution plots
    plot_files = generate_distribution_plots(data_folder)
    
    # Add plot files to artifact
    for plot_file in plot_files:
        artifact.add_file(plot_file)
    
    # Log artifact to wandb
    wandb.log_artifact(artifact)
    
    # Clean up temporary plot files
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            os.remove(plot_file)
    
    print(f"✓ Dataset artifact logged with {len(plot_files)} distribution plots")


def generate_distribution_plots(data_folder, bins=30):
    """
    Generate data distribution plots based on the presentation notebook code.
    
    Args:
        data_folder: Path to folder containing train.pt and test.pt files
        bins: Number of bins for histograms
        
    Returns:
        list: List of generated plot file paths
    """
    plot_files = []
    
    # Load data
    train_path = os.path.join(data_folder, 'train', 'train.pt')
    test_path = os.path.join(data_folder, 'test', 'test.pt')
    
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    # Extract tensors from tuples and organize by dimension
    train_tensors = [[] for _ in range(len(train_data[0]))]
    test_tensors = [[] for _ in range(len(test_data[0]))]
    
    # Collect all tensors by dimension
    for item in train_data:
        for i, tensor in enumerate(item):
            train_tensors[i].append(tensor)
    
    for item in test_data:
        for i, tensor in enumerate(item):
            test_tensors[i].append(tensor)
    
    # Stack tensors for each dimension
    train_stacked = [torch.stack(tensors) for tensors in train_tensors]
    test_stacked = [torch.stack(tensors) for tensors in test_tensors]
    
    # Create histogram plots for each tensor dimension
    n_dims = len(train_stacked)
    
    for dim_idx, (train_tensor, test_tensor) in enumerate(zip(train_stacked, test_stacked)):
        feature_dim = train_tensor.shape[1]
        
        # Create subplots for each feature in this dimension
        fig, axes = plt.subplots(2, (feature_dim + 1) // 2, figsize=(4 * ((feature_dim + 1) // 2), 8))
        if feature_dim == 1:
            axes = np.array([axes]).reshape(2, 1)
        elif feature_dim <= 2:
            axes = axes.reshape(2, -1)
        
        fig.suptitle(f'Distribution of Tensor Dimension {dim_idx} (Shape: {train_tensor.shape[1]})', fontsize=16)
        
        for feat_idx in range(feature_dim):
            row = feat_idx // ((feature_dim + 1) // 2)
            col = feat_idx % ((feature_dim + 1) // 2)
            
            if row >= axes.shape[0] or col >= axes.shape[1]:
                continue
                
            ax = axes[row, col]
            
            # Get data for this feature
            train_feature = train_tensor[:, feat_idx].numpy()
            test_feature = test_tensor[:, feat_idx].numpy()
            
            # Determine common range for both histograms
            all_data = np.concatenate([train_feature, test_feature])
            data_min, data_max = all_data.min(), all_data.max()
            bin_edges = np.linspace(data_min, data_max, bins + 1)
            
            # Plot histograms
            ax.hist(train_feature, bins=bin_edges, alpha=0.6, label=f'Train (n={len(train_feature)})', 
                   color='blue', edgecolor='black', linewidth=0.5)
            ax.hist(test_feature, bins=bin_edges, alpha=0.6, label=f'Test (n={len(test_feature)})', 
                   color='red', edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Feature {feat_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            train_mean, train_std = train_feature.mean(), train_feature.std()
            test_mean, test_std = test_feature.mean(), test_feature.std()
            stats_text = f'Train: μ={train_mean:.3f}, σ={train_std:.3f}\nTest: μ={test_mean:.3f}, σ={test_std:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        total_subplots = axes.shape[0] * axes.shape[1]
        for idx in range(feature_dim, total_subplots):
            row = idx // axes.shape[1]
            col = idx % axes.shape[1]
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"data_distribution_dim_{dim_idx}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot_file)
        plt.close()
    
    # Generate bar plot comparison
    fig, axes = plt.subplots(1, n_dims, figsize=(5*n_dims, 6))
    if n_dims == 1:
        axes = [axes]
    
    for dim_idx, (train_tensor, test_tensor) in enumerate(zip(train_stacked, test_stacked)):
        ax = axes[dim_idx]
        
        # Get the feature dimension size
        feature_dim = train_tensor.shape[1]
        
        # Calculate statistics for each feature
        train_means = train_tensor.mean(dim=0).numpy()
        test_means = test_tensor.mean(dim=0).numpy()
        train_stds = train_tensor.std(dim=0).numpy()
        test_stds = test_tensor.std(dim=0).numpy()
        
        # Create bar positions
        x = np.arange(feature_dim)
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, train_means, width, yerr=train_stds, 
                      label='Train', alpha=0.7, color='blue', capsize=5)
        bars2 = ax.bar(x + width/2, test_means, width, yerr=test_stds,
                      label='Test', alpha=0.7, color='red', capsize=5)
        
        # Customize plot
        ax.set_xlabel(f'Feature Index (Dimension {dim_idx})')
        ax.set_ylabel('Mean Value')
        ax.set_title(f'Data Distribution Comparison\n(Tensor Dimension {train_tensor.shape[1]})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in range(feature_dim)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        add_value_labels(bars1, train_means)
        add_value_labels(bars2, test_means)
    
    plt.tight_layout()
    plt.suptitle(f'Train vs Test Data Distribution Comparison\n{os.path.basename(data_folder)}', 
                 y=1.02, fontsize=14)
    
    # Save bar plot
    bar_plot_file = "data_distribution_comparison.png"
    plt.savefig(bar_plot_file, dpi=300, bbox_inches='tight')
    plot_files.append(bar_plot_file)
    plt.close()
    
    return plot_files


def train_standalone(
    dataset_path,
    success_target="score",
    preprocess_cost=False,
    hidden_layers=3,
    hidden_dim=64,
    epochs=40,
    steps_per_epoch=200,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-4,
    use_wandb=True,
    save_best_only=True,
    run_name=None,
    project_name=None
):
    """
    Standalone training on pre-existing dataset.
    
    Args:
        dataset_path (str): Path to existing dataset directory (containing metadata.json, train/, test/)
        success_target (str): Target for success metric ("error" or "score")
        preprocess_cost (bool): Apply preprocessing to cost values
        hidden_layers (int): Number of hidden layers
        hidden_dim (int): Hidden dimension size
        epochs (int): Number of training epochs
        steps_per_epoch (int): Number of steps per epoch
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        use_wandb (bool): Whether to use Weights & Biases for logging
        save_best_only (bool): Whether to save only the best checkpoint
        run_name (str): Custom run name for wandb (optional)
        project_name (str): Custom project name for wandb (optional)
        
    Returns:
        dict: Dictionary containing paths and information about trained model
    """
    print(f"Starting standalone training with dataset: {dataset_path}")
    
    # Validate dataset path
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    metadata_path = os.path.join(dataset_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")
    
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise ValueError(f"Train or test directories not found in: {dataset_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract information from dataset path if not provided
    dataset_name = os.path.basename(dataset_path)
    problem = metadata.get('problem', 'unknown')
    task = metadata.get('task', 'unknown') 
    
    # Set project and run names
    if project_name is None:
        project_name = f"{problem}_{task}"
    
    if run_name is None:
        run_name = f"{dataset_name}_h{hidden_layers}_d{hidden_dim}"
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=f"{project_name}-train",
            name=run_name
        )
    
    # Get data dimensions from metadata
    static_dim = metadata['static_dim']
    tunable_dim = metadata['tunable_dim']
    input_dim = static_dim + tunable_dim
    target_dim = metadata['result_dim']
    # success_target and preprocess_cost come from function parameters, not metadata
    
    print(f"Data dimensions - Static: {static_dim}, Tunable: {tunable_dim}, Input: {input_dim}, Target: {target_dim}")
    print(f"Success target: {success_target}, Preprocess cost: {preprocess_cost}")
    
    # Log dataset artifact with distribution plots
    #if use_wandb:
    #    print("=" * 60)
    #    print("LOGGING DATASET ARTIFACT")
    #    print("=" * 60)
    #    log_dataset_artifact_with_plots(dataset_path, project_name, run_name, metadata)
    
    # Model Training
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    config_name = f"{problem}_{task}_h{hidden_layers}_d{hidden_dim}"
    print(f"\nTraining configuration: {config_name}")
    print(f"Hidden layers: {hidden_layers}, Hidden dim: {hidden_dim}")
    
    # Create training configuration
    timestamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    
    cfg_dict = {
        'model_class': 'SimRewardModel',
        'dataset_class': 'SimRewardDataPipe', 
        'trainer_class': 'SimRewardTrainer',
        'dump_dir': f'/home/ubuntu/dev/data/surrogates/',
        'project': project_name,
        'run_name': run_name,
        'problem': problem,
        'base_seed': 42,
        'batch': batch_size,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'board': use_wandb,
        'plot': True,
        'model': {
            'n_static': static_dim,
            'n_tunable': tunable_dim,
            'input_dim': input_dim,
            'n_hidden': hidden_dim,
            'target_dim': 2,
            'hidden_layers': hidden_layers,
            'activation_mod': 'ReLU',
            'layer_norm': False,
            'res_connection': False,
            'recover_pred_unit': True,
        },
        'dataset_workers': 4,
        'dataset': {
            'dataset_root': dataset_path,
            'success_target': success_target,
            'preproc_cost': preprocess_cost
        },
        'opt': {
            'peak_lr': learning_rate,
            'weight_decay': weight_decay,
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
    
    # Convert to DictConfig
    cfg = DictConfig(cfg_dict)
    
    model_class = get_class_from_name(cfg.model_class)
    dataset_class = get_class_from_name(cfg.dataset_class)
    trainer_class = get_class_from_name(cfg.trainer_class)
    
    # Run training
    ckpt_dir = run_train(cfg, model_class, dataset_class, trainer_class)
    #ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{cfg.run_name}/best_params.pth"
    
    # Save training configuration in the same directory as best parameters
    try:
        best_params_dir = os.path.dirname(ckpt_dir)
        config_save_path = os.path.join(best_params_dir, "train_config.yaml")
        
        # Convert DictConfig to regular dict for better serialization
        cfg_dict_for_save = OmegaConf.to_container(cfg, resolve=True)
        
        # Add timestamp and additional metadata
        cfg_dict_for_save['training_completed_at'] = timestamp
        cfg_dict_for_save['best_checkpoint_path'] = ckpt_dir
        cfg_dict_for_save['data_directory'] = dataset_path
        cfg_dict_for_save['pipeline_version'] = '1.0'
        cfg_dict_for_save['standalone_script'] = True
        
        with open(config_save_path, 'w') as f:
            yaml.dump(cfg_dict_for_save, f, default_flow_style=False, indent=2)
        
        print(f"✓ Training configuration saved to: {config_save_path}")
        
    except Exception as e:
        print(f"✗ Failed to save training configuration: {str(e)}")
    
    # Run rollout on best checkpoint and log to wandb
    print("=" * 60)
    print("ROLLOUT GENERATION")
    print("=" * 60)
    
    try:
        # Create rollout configuration
        rollout_cfg = cfg.copy()
        rollout_cfg.direct_restore_dir = os.path.dirname(ckpt_dir)
        rollout_cfg.board = False  # Don't create new wandb session
        rollout_cfg.dataset.mode = "rollout"  # Set dataset to rollout mode
        
        # Run rollout (generates rollout_fig.png)
        run_rollout(rollout_cfg, model_class, dataset_class, trainer_class)
        
        # Log rollout figure to existing wandb session
        if use_wandb:
            rollout_fig_path = "/home/ubuntu/dev/src/outputs/rollout_fig.png"
            if os.path.exists(rollout_fig_path):
                wandb.log({"rollout_plot": wandb.Image(rollout_fig_path)})
                print(f"✓ Rollout figure logged to wandb: {rollout_fig_path}")
            else:
                print(f"✗ Rollout figure not found at: {rollout_fig_path}")
            
    except Exception as e:
        print(f"✗ Rollout generation failed: {str(e)}")
        
    if use_wandb:
        wandb.finish()
    
    return {
        'checkpoint_path': ckpt_dir,
        'config_path': config_save_path,
        'dataset_path': dataset_path,
        'run_name': run_name,
        'project_name': project_name
    }


def main():
    """Command line interface for the standalone training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone training script for pre-existing datasets')
    parser.add_argument('-d', '--dataset_path', type=str, required=True,
                       help='Path to existing dataset directory (e.g., /path/to/data/heat_1d/n_space_m30_pTrue)')
    parser.add_argument('-l', '--hidden_layers', type=int, default=3,
                       help='Number of hidden layers (default: 3)')
    parser.add_argument('--hidden_dims', type=int, default=64,
                       help='Hidden dimension size (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=40,
                       help='Number of training epochs (default: 40)')
    parser.add_argument('-o', '--steps_per_epoch', type=int, default=200,
                       help='Number of steps per epoch (default: 200)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('-r', '--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('-x', '--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('-n', '--run_name', type=str,
                       help='Custom run name for wandb (optional)')
    parser.add_argument('-p', '--project_name', type=str,
                       help='Custom project name for wandb (optional)')
    parser.add_argument('-s', '--success_target', type=str, default="score",
                       choices=["error", "score"],
                       help='Target for success metric ("error" or "score", default: score)')
    parser.add_argument('-c', '--preprocess_cost', action='store_true',
                       help='Apply preprocessing to cost values')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        parser.error(f"Dataset path does not exist: {args.dataset_path}")
    
    result = train_standalone(
        dataset_path=args.dataset_path,
        success_target=args.success_target,
        preprocess_cost=args.preprocess_cost,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dims,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=not args.no_wandb,
        save_best_only=True,
        run_name=args.run_name,
        project_name=args.project_name
    )
    
    print(f"\nStandalone training completed!")
    print(f"Checkpoint saved to: {result['checkpoint_path']}")
    print(f"Config saved to: {result['config_path']}")


if __name__ == "__main__":
    main()
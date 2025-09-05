from .base import Base_Trainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pdb


class SimRewardTrainer(Base_Trainer):
    """
    Trainer class for SimReward models.
    
    This trainer handles training and evaluation of reward prediction models
    using static and tunable parameters as input and reward values as targets.
    """
    
    def __init__(self, model, cfg, tc_rng):
        super().__init__(model, cfg, tc_rng)
        self.rollout_results = []
    
    def get_input_target(self, data):
        """
        Extract input and target data from the simulation reward dataset.
        
        Args:
            data: Tuple of (x_s, x_t, y) where x_s is static features,
                  x_t is tunable features, y is reward targets
                  
        Returns:
            tuple: ((x_s, x_t), y) - input tuple and target tensor
        """
        x_s, x_t, y = data
        return (x_s, x_t), y
    
    def _loss_fn(self, data):
        """
        Calculate the loss function using dimension-wise normalized MSE.

        Args:
            data: PyTorch Geometric Data object containing input and target data.

        Returns:
            torch.Tensor: Normalized MSE loss value.
        """
        pred = self.get_pred(data)
        _, target = self.get_input_target(data)
        
        # Get target normalizer statistics for dimension-wise scaling
        model = self.model.module if hasattr(self.model, "module") else self.model
        target_normalizer = model._targetNormalizer
        target_std = target_normalizer.std_with_epsilon()
        
        # Dimension-wise normalized MSE
        normalized_error = (pred - target) / (target_std + 1e-8)

        return torch.mean(normalized_error ** 2)
    
    def accumulate(self, data):
        """
        Accumulate statistics for model normalizers using sim reward data.
        
        Args:
            data: Tuple of (x_s, x_t, y)
        """
        data = self.move_to_device(data)
        input_tuple, target = self.get_input_target(data)
        x_s, x_t = input_tuple
        
        # Concatenate input features for normalization (matching model preprocessing)
        concatenated_input = torch.cat([x_s, x_t], dim=-1)
        try:
            self.model.accumulate(concatenated_input, target)
        except:
            self.model.module.accumulate(concatenated_input, target)
    
    def get_metrics(self, data):
        """
        Calculate metrics for reward prediction.
        
        Args:
            data: Tuple of (x_s, x_t, y)
            
        Returns:
            tuple: (mean_absolute_error, rmse, r2_score)
        """
        data = self.move_to_device(data)
        pred = self.get_pred(data)
        _, target = self.get_input_target(data)
        
        model = self.model.module if hasattr(self.model, "module") else self.model
        target_normalizer = model._targetNormalizer
        target_std = target_normalizer.std_with_epsilon()

        normalized_error = (pred - target) / (target_std + 1e-8)
        
        # Calculate various metrics
        mae = torch.mean(torch.abs(normalized_error))
        mse = torch.mean((normalized_error) ** 2)
        rmse = torch.sqrt(mse)
        
        # R� score
        mean_error = torch.mean(normalized_error)
        ss_tot = torch.sum((mean_error) ** 2)
        ss_res = torch.sum((mean_error) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return mae.item(), rmse.item(), r2.item()
    
    def print_metrics(self, data, prefix):
        """
        Print reward prediction metrics.
        
        Args:
            data: Tuple of (x_s, x_t, y)
            prefix: String prefix for output (e.g., 'train', 'test')
        """
        mae, rmse, r2 = self.get_metrics(data)
        
        print(f"train step: {self.train_step}, {prefix}_MAE: {mae:.6f}, "
              f"{prefix}_RMSE: {rmse:.6f}, {prefix}_R�: {r2:.6f}")
        
        # Skip wandb logging for now
        try:
            if wandb.run is not None:
                wandb.log({
                    "step": self.train_step,
                    f"{prefix}_MAE": mae,
                    f"{prefix}_RMSE": rmse,
                    f"{prefix}_R2": r2
                })
        except AttributeError:
            pass
    
    def eval_plot(self, data=None, prefix="rollout", board=False, precomputed=False, pred_np=None, target_np=None, 
                 pred_s_np=None, pred_c_np=None, target_s_np=None, target_c_np=None):
        """
        Create evaluation plots for reward prediction.
        
        Args:
            data: Tuple of (x_s, x_t, y)
            prefix: String prefix for plot titles
            board: Whether to log to wandb
            precomputed: Whether to use precomputed values
            pred_s_np, pred_c_np: Predicted success and cost arrays
            target_s_np, target_c_np: Target success and cost arrays
        """
        if not precomputed:
            data = self.move_to_device(data)
            pred = self.get_pred(data)
            _, target = self.get_input_target(data)
        
            # Convert to numpy for plotting and separate dimensions
            pred_np = pred.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            pred_s_np = pred_np[:, 0]  # success dimension
            pred_c_np = pred_np[:, 1]  # cost dimension
            target_s_np = target_np[:, 0]  # success dimension
            target_c_np = target_np[:, 1]  # cost dimension
        
        #pdb.set_trace()
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Success prediction vs target scatter
        plt.subplot(1, 3, 1)
        plt.scatter(target_s_np, pred_s_np, alpha=0.6, s=10)
        plt.plot([target_s_np.min(), target_s_np.max()], [target_s_np.min(), target_s_np.max()], 'r--', lw=2)
        plt.xlim(min(target_s_np.min(), pred_s_np.min()), 
            max(target_s_np.max(), pred_s_np.max()))
        plt.ylim(min(target_s_np.min(), pred_s_np.min()), 
            max(target_s_np.max(), pred_s_np.max()))
        plt.xlabel('True Success')
        plt.ylabel('Predicted Success')
        plt.title(f'{prefix} - Success Prediction vs True')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Cost prediction vs target scatter
        plt.subplot(1, 3, 2)
        plt.scatter(target_c_np, pred_c_np, alpha=0.6, s=10)
        plt.plot([target_c_np.min(), target_c_np.max()], [target_c_np.min(), target_c_np.max()], 'r--', lw=2)
        plt.xlim(min(target_c_np.min(), pred_c_np.min()), 
            max(target_c_np.max(), pred_c_np.max()))
        plt.ylim(min(target_c_np.min(), pred_c_np.min()), 
            max(target_c_np.max(), pred_c_np.max()))
        plt.xlabel('True Cost')
        plt.ylabel('Predicted Cost')
        plt.title(f'{prefix} - Cost Prediction vs True')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Success vs Cost scatter plot (replacing residual plot)
        plt.subplot(1, 3, 3)
        # Plot predictions in one color and targets in another
        plt.scatter(pred_s_np, pred_c_np, alpha=0.6, s=10, label='Predictions', color='blue')
        plt.scatter(target_s_np, target_c_np, alpha=0.6, s=10, label='Targets', color='red')
        plt.xlabel('Success')
        plt.ylabel('Cost')
        plt.title(f'{prefix} - Success vs Cost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if board and wandb.run is not None:
            wandb.log({f"{prefix}_eval_plot": wandb.Image(plt)})
        
        plt.savefig('/home/ubuntu/dev/src/outputs/rollout_fig.png')
        plt.close()
    
    def rollout(self, data):
        """
        Alias for run_rollout to match the interface expected by run_rollout.py
        """
        return self.run_rollout(data)
    
    def run_rollout(self, data):
        """
        Run rollout for reward prediction (simple forward pass).
        
        Args:
            data: Tuple of (x_s, x_t, y)
            
        Returns:
            dict: Dictionary containing predictions and targets
        """
        data = self.move_to_device(data)
        pred = self.get_pred(data)
        _, target = self.get_input_target(data)

        model = self.model.module if hasattr(self.model, "module") else self.model
        target_normalizer = model._targetNormalizer
        target_std = target_normalizer.std_with_epsilon()

        normalized_error = (pred - target) / (target_std + 1e-8)
        #pdb.set_trace()
        return {
            'predictions': pred.cpu().detach(),
            'targets': target.cpu().detach(),
            'mae': torch.mean(torch.abs(normalized_error)).item(),
            'rmse': torch.sqrt(torch.mean(normalized_error ** 2)).item()
        }
    
    def post_process_rollout(self, data, rollout_res, bi):
        """
        Post-process rollout results and store them.
        
        Args:
            data: Input data
            rollout_res: Results from run_rollout
            bi: Batch index
        """
        self.rollout_results.append({
            'batch_idx': bi,
            'mae': rollout_res['mae'],
            'rmse': rollout_res['rmse'],
            'predictions': rollout_res['predictions'],
            'targets': rollout_res['targets']
        })
    
    def summarize_rollout(self):
        """
        Summarize all rollout results and print statistics.
        """
        if not self.rollout_results:
            print("No rollout results to summarize.")
            return
        
        all_maes = [res['mae'] for res in self.rollout_results]
        all_rmses = [res['rmse'] for res in self.rollout_results]
        
        # Collect all predictions and targets
        all_preds = torch.cat([res['predictions'] for res in self.rollout_results])
        all_targets = torch.cat([res['targets'] for res in self.rollout_results])
        
        # Overall metrics
        overall_mae = np.mean(all_maes)
        overall_rmse = np.mean(all_rmses)
        
        # Calculate R�
        target_mean = torch.mean(all_targets)
        ss_tot = torch.sum((all_targets - target_mean) ** 2)
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        print("\n" + "="*50)
        print("ROLLOUT SUMMARY")
        print("="*50)
        print(f"Total batches: {len(self.rollout_results)}")
        print(f"Total samples: {len(all_preds)}")
        print(f"Overall MAE: {overall_mae:.6f}")
        print(f"Overall RMSE: {overall_rmse:.6f}")
        print(f"Overall R�: {r2.item():.6f}")
        print(f"MAE std: {np.std(all_maes):.6f}")
        print(f"RMSE std: {np.std(all_rmses):.6f}")
        print("="*50)
        
        # Skip wandb logging for rollout summary
        # Log to wandb if available
        try:
            if wandb.run is not None:
                wandb.log({
                    "rollout_overall_MAE": overall_mae,
                    "rollout_overall_RMSE": overall_rmse,
                    "rollout_overall_R2": r2.item(),
                    "rollout_MAE_std": np.std(all_maes),
                    "rollout_RMSE_std": np.std(all_rmses)
                })
        except AttributeError:
            pass
        
        # Clear results for next rollout
        self.rollout_results = []
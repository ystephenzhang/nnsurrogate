import os, sys
import torch
import hydra
import wandb
import pytz
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
sys.path.append('/home/ubuntu/dev/src/')
from template_trainer.utils import (
    set_seed,
    timer,
    InfiniteDataLooper,
)

from tqdm import tqdm
from template_trainer.model.sim import SimRewardModel
from template_trainer.dataset.sim import SimRewardDataPipe
from template_trainer.trainer.sim import SimRewardTrainer
import torch.nn as nn

import pdb

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_class_from_name(class_name):
    """Resolve class name string to actual class object."""
    class_map = {
        'SimRewardModel': SimRewardModel,
        'SimRewardDataPipe': SimRewardDataPipe,
        'SimRewardTrainer': SimRewardTrainer,
    }
    return class_map[class_name]


def get_activation_module(activation_name):
    """Resolve activation name string to actual activation module instance."""
    activation_map = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'LeakyReLU': nn.LeakyReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
    }
    return activation_map[activation_name]


def run_train(cfg, model_class, dataset_class, trainer_class):
    """
    Run the training loop.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    '''if cfg.board:
        wandb.init(
            project=f"{cfg.project}-train",
            config=OmegaConf.to_container(cfg, resolve=True),
            name = cfg.runname
        )'''

    # Model and dataset creation
    model = model_class(cfg.model)
    train_datapipe = dataset_class(cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train")
    test_datapipe = dataset_class(cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")

    # Trainer creation
    trainer = trainer_class(model, cfg, tc_rng)

    # Data loaders creation
    train_loader = DataLoader(
        train_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    # Infinite data loopers for training and testing
    train_loopers = InfiniteDataLooper(train_loader)
    test_loopers = InfiniteDataLooper(test_loader)

    # Best checkpoint tracking
    best_test_loss = float('inf')
    best_checkpoint_path = None

    # Accumulate channel mean and std for model
    accumulation_steps = cfg.opt.accumulation_steps
    print("Accumulating channel mean and std for model...")
    for _ in tqdm(range(accumulation_steps)):
        data = next(train_loopers)
        #pdb.set_trace()
        trainer.accumulate(data)
    print("Accumulation done. The stats are:")
    try:
        trainer.model.report_stats()
    except:
        trainer.model.module.report_stats()

    # Training loop starts
    total_steps = cfg.epochs * cfg.steps_per_epoch
    print("Training starts...")
    for _ in range(total_steps + 1):
        train_data = next(train_loopers)

        # Log loss
        if (
            (trainer.train_step % cfg.loss_freq == 0)
            or (trainer.train_step % (cfg.loss_freq) == 0 and trainer.train_step <= cfg.loss_freq)
            or (trainer.train_step % (cfg.loss_freq) == 0 and trainer.train_step >= total_steps - cfg.loss_freq)
        ):
            with torch.no_grad():
                # Train loss and error
                trainer.print_metrics(train_data, "train")
                trainer.board_loss(train_data, "train", cfg.board)

                # Test loss and error
                test_data = next(test_loopers)
                trainer.print_metrics(test_data, "test")
                trainer.board_loss(test_data, "test", cfg.board)
                
                # Check if current test loss is the best so far
                current_test_loss = trainer.get_loss(test_data).item()
                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    # Save best checkpoint
                    ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{cfg.run_name}"
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    
                    # Remove previous best checkpoint if it exists
                    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                        os.remove(best_checkpoint_path)
                    
                    # Save new best checkpoint
                    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                    best_checkpoint_path = f"{ckpt_dir}/best_params.pth"
                    torch.save(model.state_dict(), best_checkpoint_path)
                    print(f"New best checkpoint saved: {best_checkpoint_path} with test loss: {best_test_loss:.6f}")

        # Log test error plot
        if cfg.plot and (
            (trainer.train_step % cfg.plot_freq == 0)
            or (trainer.train_step % (cfg.plot_freq) == 0 and trainer.train_step <= cfg.plot_freq)
            or (trainer.train_step % (cfg.plot_freq) == 0 and trainer.train_step >= total_steps - cfg.plot_freq)
        ):
            test_data = next(test_loopers)
            trainer.eval_plot(test_data, "test", cfg.board)

        # Save checkpoint
        if trainer.train_step % cfg.save_freq == 0:
            ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            print("Current time: " + datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S"))
            trainer.save(ckpt_dir)

        # Training iteration
        trainer.iter(train_data)

        # Time estimation
        if trainer.train_step == cfg.time_warm:
            timer.tic("time estimate")
        if (
            trainer.train_step > 0
            and ((trainer.train_step - cfg.time_warm) % cfg.time_freq == 0)
            and trainer.train_step - cfg.time_warm > 0
        ):
            ratio = (trainer.train_step - cfg.time_warm) / total_steps
            timer.estimate_time("time estimate", ratio)

    # Print final best checkpoint info
    if best_checkpoint_path:
        print(f"\nTraining completed. Best checkpoint saved at: {best_checkpoint_path}")
        print(f"Best test loss: {best_test_loss:.6f}")
    else:
        print("\nTraining completed. No best checkpoint was saved (no test loss evaluation occurred).")
    
    '''if cfg.board:
        wandb.finish()'''
    
    return best_checkpoint_path


# =====================================================================
# An example for running with hydra configs
# =====================================================================
@hydra.main(version_base=None, config_path="train_configs", config_name="1D_heat_transfer.yaml")
def main(cfg):
    """
    Main function to run the training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    model_class = get_class_from_name(cfg.model_class)
    dataset_class = get_class_from_name(cfg.dataset_class)
    trainer_class = get_class_from_name(cfg.trainer_class)
    run_train(cfg, model_class, dataset_class, trainer_class)


if __name__ == "__main__":
    main()

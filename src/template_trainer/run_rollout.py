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
)
import pdb
from tqdm import tqdm
from template_trainer.model.sim import SimRewardModel
from template_trainer.dataset.sim import SimRewardDataPipe
from template_trainer.trainer.sim import SimRewardTrainer
import torch.nn as nn
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def get_class_from_name(class_name):
    """Resolve class name string to actual class object."""
    class_map = {
        'SimRewardModel': SimRewardModel,
        'SimRewardDataPipe': SimRewardDataPipe,
        'SimRewardTrainer': SimRewardTrainer,
    }
    return class_map[class_name]
@torch.no_grad()
def run_rollout(cfg, model_class, dataset_class, trainer_class):
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

    if cfg.board:
        wandb.init(
            project=f"{cfg.project}-train",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Model and dataset creation
    model = model_class(cfg.model)

    # Trainer creation
    trainer = trainer_class(model, cfg, tc_rng)
    # restore model
    trainer.restore(cfg.direct_restore_dir)

    # Data loaders creation
    test_datapipe = dataset_class(cfg.dataset, 0, cfg.base_seed, "rollout")
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=0,
        pin_memory=True,
    )
    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    pred_s = []  # success predictions (dimension 0)
    pred_c = []  # cost predictions (dimension 1)
    target_s = []  # success targets (dimension 0)
    target_c = []  # cost targets (dimension 1)
    # Rollout loop starts
    for bi, batch in tqdm(enumerate(test_loader)):
        # to device
        batch = trainer.move_to_device(batch)
        # run rollout for 1 batch
        rollout_res = trainer.rollout(batch)
        # post-process the result
        
        # Extract predictions and targets for each dimension
        predictions = rollout_res["predictions"]
        targets = rollout_res["targets"]

        #pdb.set_trace()
        
        # Separate dimensions: 0=success, 1=cost
        pred_s.extend(predictions[:, 0].cpu().detach().numpy())
        pred_c.extend(predictions[:, 1].cpu().detach().numpy())
        target_s.extend(targets[:, 0].cpu().detach().numpy())
        target_c.extend(targets[:, 1].cpu().detach().numpy())
        
        trainer.post_process_rollout(batch, rollout_res, bi)
    
    # Convert lists to numpy arrays for plotting
    import numpy as np
    pred_s_np = np.array(pred_s)
    pred_c_np = np.array(pred_c)
    target_s_np = np.array(target_s)
    target_c_np = np.array(target_c)
    
    trainer.eval_plot(prefix="test_rollout", precomputed=True, 
                     pred_s_np=pred_s_np, pred_c_np=pred_c_np, 
                     target_s_np=target_s_np, target_c_np=target_c_np)

    # summarize results
    trainer.summarize_rollout()

    if cfg.board:
        wandb.finish()

@hydra.main(version_base=None, config_path="/home/ubuntu/dev/src/template_trainer/train_configs", config_name="1D_heat_transfer.yaml")
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
    run_rollout(cfg, model_class, dataset_class, trainer_class)

if __name__ == "__main__":
    main()
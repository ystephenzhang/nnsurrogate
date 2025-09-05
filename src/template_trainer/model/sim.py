from .base import Base
import torch.nn as nn
import torch
from ..ops.basic import MLP
import pdb

class SimRewardModel(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.preprocessed_output = cfg.preprocessed_output
        self.n_static = cfg.n_static
        self.n_tunable = cfg.n_tunable
        self.n_hidden = cfg.n_hidden
        self.n_target = cfg.target_dim
        
        # Branch 1 — Static parameters
        self.static_fc1 = nn.Linear(self.n_static, self.n_hidden)
        self.static_fc2 = nn.Linear(self.n_hidden, self.n_hidden)

        # Branch 2 — Tunable parameters
        self.tunable_fc1 = nn.Linear(self.n_tunable, self.n_hidden)
        self.tunable_fc2 = nn.Linear(self.n_hidden, self.n_hidden)

        self.mlp = MLP((2 * self.n_hidden),
                       self.n_hidden, 
                       self.n_target, 
                       cfg.hidden_layers,
                       layer_normalized=cfg.layer_norm,
                       res_connection=cfg.res_connection)
    
    def _forward(self, input, **kwargs):
        """
        Forward pass through the reward model.
        
        Args:
            input: Normalized concatenated tensor of [x_s, x_t] features
                   
        Returns:
            torch.Tensor: Predicted reward values
        """
        import torch
        
        # Split the concatenated normalized input back into static and tunable parts
        x_s = input[:, :self.n_static]
        x_t = input[:, self.n_static:]
        
        # Process static features through first branch
        static_out = torch.relu(self.static_fc1(x_s))
        static_out = torch.relu(self.static_fc2(static_out))
        # Process tunable features through second branch  
        tunable_out = torch.relu(self.tunable_fc1(x_t))
        tunable_out = torch.relu(self.tunable_fc2(tunable_out))
        # Concatenate processed features
        combined = torch.cat([static_out, tunable_out], dim=-1)
        
        # Pass through final MLP to get predictions
        output = self.mlp(combined)
        return output
    
    def _preprocess(self, input, **kwargs):
        x_s, x_t = input
        normalized_input = self._inputNormalizer(torch.cat([x_s, x_t], dim=-1), accumulate=False)
        return normalized_input
    
    def forward(self, input, **kwargs):
        """
        Perform the forward pass of the model with normalization.

        This method normalizes the input, calls the model-specific _forward method,
        and then denormalizes the output.

        Args:
            input (torch.Tensor): Input tensor of shape (B, ..., C_in).

        Returns:
            torch.Tensor: Denormalized output tensor of shape (B, ..., C_out).

        Process:
            input -> normalize -> model -> denormalize -> output
        """
        _, x_t = input
        input = self._preprocess(input, **kwargs)
        pred = self._forward(input, **kwargs)
        pred = self._postprocess(pred, **kwargs)

        if self.preprocessed_output:
            pred[:, 1] = pred[:, 1] * x_t[:, 0] * x_t[:, 0] / x_t[:, 1]
        return pred
        
"""Uncertainty-Aware Prediction with Gradient Descent Decoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyAwarePGD(nn.Module):
    """
    Uncertainty-Aware Prediction with Gradient Descent
    - Decodes trajectory features + intent anchor into future displacement
    - Outputs both prediction and log-variance for uncertainty
    """
    def __init__(self, d_model: int = 256, pred_len: int = 20, trajectory_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.trajectory_dim = trajectory_dim

        # Feature aggregation
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Trajectory decoder: produces displacement delta at each step
        self.delta_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * 3)  # (B, pred_len*3)
        )

        # Uncertainty head: log-variance for each prediction step
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, pred_len * 3)
        )

    def forward(
        self,
        encoded_feat: torch.Tensor,
        global_anchor: torch.Tensor,
        historical_trajectory: torch.Tensor = None,
        return_uncertainty: bool = True
    ):
        """
        Args:
            encoded_feat: (B, T, d_model) encoded trajectory
            global_anchor: (B, 1, d_model) intent anchor
            historical_trajectory: (B, T, 6) raw trajectory (for baseline)
            return_uncertainty: whether to output logvar
        Returns:
            dict with:
                predictions: (B, pred_len, 3) future displacement
                logvar: (B, pred_len, 3) log variance for uncertainty
        """
        B, T, D = encoded_feat.shape

        # Aggregate trajectory features
        traj_feat = encoded_feat.mean(dim=1)  # (B, d_model)
        # Combine with intent anchor
        anchor_feat = global_anchor.squeeze(1)  # (B, d_model)
        combined = torch.cat([traj_feat, anchor_feat], dim=-1)  # (B, d_model*2)
        fused = self.feature_proj(combined)  # (B, d_model)

        # Decode displacement
        delta_flat = self.delta_decoder(fused)  # (B, pred_len*3)
        predictions = delta_flat.view(B, self.pred_len, 3)  # (B, pred_len, 3)

        # Uncertainty
        if return_uncertainty:
            logvar_flat = self.uncertainty_head(fused)  # (B, pred_len*3)
            logvar = logvar_flat.view(B, self.pred_len, 3)  # (B, pred_len, 3)
        else:
            logvar = torch.zeros_like(predictions)

        return {
            'predictions': predictions,
            'logvar': logvar,
        }

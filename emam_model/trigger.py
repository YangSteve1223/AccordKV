"""Event-Driven Trigger for Adaptive Trajectory Prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTrigger(nn.Module):
    """Fallback simple trigger: always fires"""
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, trajectory, intent_logits, intent_history=None):
        B = trajectory.shape[0]
        return {
            'trigger_decision': torch.ones(B, dtype=torch.bool, device=trajectory.device),
            'trigger_score': torch.ones(B, device=trajectory.device),
            'maneuver_score': torch.zeros(B, device=trajectory.device),
        }


class EventDrivenTrigger(nn.Module):
    """
    Event-Driven Trigger
    Decides whether to trigger full trajectory prediction
    based on trajectory dynamics + intent signal
    """
    def __init__(
        self,
        feature_dim: int = 6,
        num_intent_classes: int = 5,
        threat_weight: float = 0.3,
        intent_weight: float = 0.3,
        spatial_weight: float = 0.4
    ):
        super().__init__()
        self.threat_weight = threat_weight
        self.intent_weight = intent_weight
        self.spatial_weight = spatial_weight

        # Trajectory-based threat assessment: velocity at last timestep
        self.threat_scorer = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Intent-based urgency
        self.intent_scorer = nn.Sequential(
            nn.Linear(num_intent_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Spatial dynamics: aggregate over trajectory
        self.spatial_scorer = nn.Sequential(
            nn.Linear(8, 64),  # mean_vel(3) + max_speed(1) + vel_std(3) + speed_change(1)
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectory: torch.Tensor, intent_logits: torch.Tensor, intent_history=None):
        """
        Args:
            trajectory: (B, T, feature_dim) recent trajectory
            intent_logits: (B, num_classes) current intent probs
            intent_history: (T, num_classes) optional historical intent
        Returns:
            dict with:
                trigger_decision: (B,) bool
                trigger_score: (B,) float [0,1]
                maneuver_score: (B,) float [0,1]
        """
        B, T, C = trajectory.shape

        # Threat score: based on velocity at last timestep
        vel = trajectory[:, -1, 3:6]  # (B, 3)
        threat = self.threat_scorer(vel).squeeze(-1)  # (B,)

        # Intent urgency
        intent_probs = F.softmax(intent_logits, dim=-1)
        intent_score = self.intent_scorer(intent_probs).squeeze(-1)  # (B,)

        # Spatial dynamics score: aggregate stats over trajectory
        all_vel = trajectory[:, :, 3:6]  # (B, T, 3)
        speed = torch.norm(all_vel, dim=-1)  # (B, T)
        mean_vel = all_vel.mean(dim=1)  # (B, 3)
        max_speed = speed.max(dim=1, keepdim=True)[0]  # (B, 1)
        vel_std = all_vel.std(dim=1)  # (B, 3)
        speed_change = speed[:, -1:] - speed[:, :1].mean(dim=1, keepdim=True)  # (B, 1)
        spatial_input = torch.cat([mean_vel, max_speed, vel_std, speed_change], dim=-1)  # (B, 8)
        spatial = self.spatial_scorer(spatial_input).squeeze(-1)  # (B,)

        # Weighted combination
        trigger_score = (
            self.threat_weight * threat +
            self.intent_weight * intent_score +
            self.spatial_weight * spatial
        )

        trigger_decision = trigger_score > 0.5

        # Maneuver score: deviation from straight-line
        speed = torch.norm(all_vel, dim=-1)  # (B, T)
        maneuver_score = speed.std(dim=1) / (speed.mean(dim=1) + 1e-8)

        return {
            'trigger_decision': trigger_decision,
            'trigger_score': trigger_score,
            'maneuver_score': maneuver_score,
        }

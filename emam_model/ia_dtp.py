"""Intent-Aware Destination-Trajectory Prediction Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Intent type enumeration
class IntentType:
    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    ASCEND = 3
    DESCEND = 4


NUM_INTENT_CLASSES = 5


class IntentAwareDTP(nn.Module):
    """
    Intent-Aware Destination Trajectory Prediction
    - Classifies maneuver intent from encoded trajectory features
    - Produces intent-conditioned global anchor for decoder
    """
    def __init__(self, d_model: int = 256, num_classes: int = NUM_INTENT_CLASSES, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes

        # Intent classification head
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # Global anchor generator (learned intent-conditioned destination)
        self.anchor_proj = nn.Linear(d_model + num_classes, d_model)

        # Feature enhancement: blend intent info back into trajectory features
        self.enhance = nn.Sequential(
            nn.Linear(d_model + num_classes, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, encoded_features: torch.Tensor, historical_trajectory: torch.Tensor = None):
        """
        Args:
            encoded_features: (B, T, d_model) trajectory encoded features
            historical_trajectory: (B, T, 6) raw trajectory (optional)
        Returns:
            dict with:
                global_anchor: (B, 1, d_model) intent-conditioned anchor
                intent_logits: (B, num_classes) intent logits
                intent_weights: (B, num_classes) softmax intent weights
                enhanced_features: (B, T, d_model) intent-enhanced features
        """
        B, T, D = encoded_features.shape

        # Pool to global representation
        global_feat = encoded_features.mean(dim=1)  # (B, d_model)

        # Intent classification
        intent_logits = self.intent_head(global_feat)  # (B, num_classes)
        intent_probs = F.softmax(intent_logits, dim=-1)  # (B, num_classes)
        intent_weights = intent_probs  # alias for clarity

        # Intent-conditioned anchor
        intent_context = intent_probs.unsqueeze(1)  # (B, 1, num_classes)
        global_expanded = global_feat.unsqueeze(1)  # (B, 1, d_model)
        anchor_input = torch.cat([global_expanded, intent_context], dim=-1)  # (B, 1, d_model+num_classes)
        global_anchor = self.anchor_proj(anchor_input)  # (B, 1, d_model)

        # Enhanced features (blend intent info into each timestep)
        intent_tiled = intent_probs.unsqueeze(1).expand(-1, T, -1)  # (B, T, num_classes)
        enhance_input = torch.cat([encoded_features, intent_tiled], dim=-1)
        enhanced_features = self.enhance(enhance_input)  # (B, T, d_model)

        return {
            'global_anchor': global_anchor,
            'intent_logits': intent_logits,
            'intent_weights': intent_weights,
            'enhanced_features': enhanced_features,
        }

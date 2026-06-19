"""
Enhanced Mamba with Squeeze-and-Excitation (EMam-SE)
Input: Trajectory Sequence → Spatiotemporal Encoded Features
1. Linear Projection: (B,T,6) → (B,T,D)
2. LayerNorm
3. 1D Causal Conv (kernel=3)
4. EnhancedMambaBlock × n_layers
5. MultiScaleDWConv (3/5/7)
6. SE Channel Attention
7. Gate + LayerNorm + Dropout
8. Output Projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit): x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu(x)


class GRLU(nn.Module):
    """Gated Recurrent Linear Unit"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba variant)
    Input: (B,T,D) → (B,T,D)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 3, bias=False)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.ones_(self.D)
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        xz = self.in_proj(x)
        x_gate, x_dt_raw, x_inner = xz.chunk(3, dim=-1)
        x_gate = torch.sigmoid(x_gate)
        x_inner = x_inner * x_gate

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_conv = silu(x_conv)

        dt = F.softplus(x_dt_raw)

        A_dis = torch.exp(torch.clamp(self.A, min=-50, max=50))  # (d_inner, d_state)
        h = torch.einsum('bti,ik->bti', torch.cumsum(dt * x_conv, dim=1), A_dis)
        y = h + self.D.unsqueeze(0).unsqueeze(0)

        output = self.out_proj(y)
        return output


class MambaBlock(nn.Module):
    """Mamba Block with Gated Linear Unit"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.grlu = GRLU(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.grlu(x)
        x = self.dropout(x)
        return residual + x


class MultiScaleDWConv1D(nn.Module):
    """
    Multi-Scale Depthwise Conv (kernel=3,5,7)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv_3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv_5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.fusion = nn.Conv1d(dim * 3, dim, kernel_size=1)
        self._init()

    def _init(self):
        for conv in [self.conv_3, self.conv_5, self.conv_7]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        o3 = self.conv_3(x)
        o5 = self.conv_5(x)
        o7 = self.conv_7(x)
        out = torch.cat([o3, o5, o7], dim=1)
        out = self.fusion(out)
        return out.transpose(1, 2)


class SEChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            SiLU(),
            nn.Linear(dim // reduction, dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        w = self.squeeze(x_t).squeeze(-1)
        w = self.excitation(w)
        return x * w.unsqueeze(1)


class EnhancedMambaSE(nn.Module):
    """
    Enhanced Mamba with SE (EMam-SE)
    1. Linear Projection: (B,T,6) → (B,T,D)
    2. LayerNorm
    3. Causal Conv (kernel=3)
    4. Mamba blocks + MultiScaleDWConv + SE
    5. Final gating + dropout
    6. Output Projection
    """
    def __init__(self, input_dim: int = 6, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.causal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.ms_dwconv = MultiScaleDWConv1D(d_model)
        self.se_attention = SEChannelAttention(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.output_proj = nn.Linear(d_model, d_model)
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)
        x_conv = self.causal_conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = self.act(x_conv)
        x = x + x_conv

        for block in self.mamba_blocks:
            x = block(x)
            if hasattr(self, 'ms_dwconv'):
                x_ms = self.ms_dwconv(x)
                x = x + x_ms
                x = self.se_attention(x)

        gate = torch.sigmoid(self.gate_proj(x))
        x = self.norm(x)
        x = gate * x
        x = self.dropout(x)
        return self.output_proj(x)


class EMAMSEWithOutput(nn.Module):
    """EMam-SE with output pooling"""
    def __init__(self, input_dim: int = 6, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.core = EnhancedMambaSE(input_dim, d_model, d_state, d_conv, expand, n_layers, dropout)

    def forward(self, x):
        features = self.core(x)
        pooled = features.mean(dim=1)
        return features, pooled

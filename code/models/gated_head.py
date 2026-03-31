import torch
import torch.nn as nn


class GatedDualBranchHead(nn.Module):
    """
    Gated Dual-Branch Regression Head for long-tail SSC estimation.
    
    This head employs a Mixture-of-Experts (MoE) architecture with a gating
    mechanism to dynamically route features between specialized head (low-SSC)
    and tail (high-SSC) experts.
    
    The final prediction is computed as:
        y_hat = g * M_head(F_pool) + (1 - g) * M_tail(F_pool)
    where g = sigmoid(w_g^T * F_pool) is the gating weight.
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_classes=1, dropout=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Head expert: optimized for low-concentration regimes
        self.head_expert = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Tail expert: designed for high-concentration (hyper-concentrated) flows
        self.tail_expert = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Gating network: dynamically adjusts the weight between head and tail experts
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Pooled visual feature vector of shape (B, in_dim).
        
        Returns:
            Tensor: Predicted SSC values of shape (B, num_classes).
            Tensor: Gating weight g of shape (B, 1).
            Tensor: Head expert output of shape (B, num_classes).
            Tensor: Tail expert output of shape (B, num_classes).
        """
        # Compute expert outputs
        head_out = self.head_expert(x)   # (B, num_classes)
        tail_out = self.tail_expert(x)   # (B, num_classes)
        
        # Compute gating weight
        g = self.gate(x)  # (B, 1)
        
        # Dynamic weighted ensemble
        y_hat = g * head_out + (1.0 - g) * tail_out  # (B, num_classes)
        
        return y_hat, g, head_out, tail_out

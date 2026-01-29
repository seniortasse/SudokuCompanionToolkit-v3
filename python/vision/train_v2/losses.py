from __future__ import annotations
import torch
import torch.nn as nn

def ce_with_class_weights(num_classes: int = 10, w0: float = 0.25, w_pos: float = 1.0, device="cpu"):
    """
    Class-weighted CE for heads with class 0 = 'no mark'.
    """
    w = torch.full((num_classes,), float(w_pos), dtype=torch.float32, device=device)
    w[0] = float(w0)
    return nn.CrossEntropyLoss(weight=w)

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Multi-label focal BCE.
    gamma: focusing parameter
    pos_weight: like BCEWithLogitsLoss's pos_weight (tensor or float)
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float | torch.Tensor = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if isinstance(pos_weight, (int, float)):
            self.pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32)
        else:
            self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits to get base loss per element
        # We'll reconstruct focal weighting manually.
        # Shapes: [B, C]
        pos_w = self.pos_weight.to(logits.device)
        if pos_w.ndim == 1 and pos_w.numel() == 1:
            pos_w = pos_w.expand_as(logits)

        # Standard BCE with logits (no reduction)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_w, reduction="none"
        )

        # p_t = sigmoid(logit) if y=1 else 1 - sigmoid(logit)
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)

        focal = (1 - p_t) ** self.gamma
        loss = focal * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def build_losses(
    device: torch.device,
    w0_given: float = 0.25,
    w0_solution: float = 0.25,
    cand_pos_weight: float = 4.0,
    focal_gamma: float = 2.0,
):
    ce_given = ce_with_class_weights(10, w0=w0_given, w_pos=1.0, device=device)
    ce_solution = ce_with_class_weights(10, w0=w0_solution, w_pos=1.0, device=device)
    bce_cand = FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=cand_pos_weight, reduction="mean")
    return ce_given, ce_solution, bce_cand
# =============================================================================
# CellNet — Multi‑Head Sudoku Cell Interpreter (FINAL, fully annotated)
# =============================================================================
# PURPOSE & STORY — WHY THIS MODEL EXISTS
# -----------------------------------------------------------------------------
# In the Sudoku Companion, every 9×9 board is decomposed into 81 cell crops. For
# each cell, we want three complementary predictions:
#   1) GIVEN (single‑label 0–9): what printed digit (if any) is visibly present?
#   2) SOLUTION (single‑label 0–9): what is the solved digit for this cell?
#   3) CANDIDATES (multi‑label 0–9): which digits are plausible pencil‑marks?
#
# These three tasks share a common visual understanding of the cell (edges,
# ink/print, handwriting strokes, noise, etc.). So we use a SHARED CONVOLUTIONAL
# TRUNK that extracts general features once, then branch into three HEADS:
#   - Head #1 (Given):      global, classification → 10 logits (softmax at loss)
#   - Head #2 (Solution):   global, classification → 10 logits (softmax at loss)
#   - Head #3 (Candidates): spatial, multi‑label   → 10 heatmaps + pooled logits
#
# DESIGN PHILOSOPHY
# -----------------------------------------------------------------------------
# • A single, light backbone (few conv layers) keeps runtime small on‑device.
# • Global heads (Given/Solution) use GAP→Linear: robust and inexpensive.
# • Candidate head is SPATIAL and DEEPER with dilation: we want to see strokes
#   and micro‑patterns anywhere in the cell, but also aggregate them smoothly.
#   We therefore:
#     (1) produce per‑digit FEATURE MAPS ("cand_maps": [B,10,h,w]) at 1/4 res,
#     (2) pool them with LogSumExp (LSE) at temperature τ (lse_tau), which is a
#         smooth approximation to global‑max. τ↓ behaves more like max‑pooling;
#         τ↑ becomes more average‑like. We empirically use τ≈0.40.
# • We RETURN both the pooled candidate logits and the raw cand_maps so that the
#   training pipeline can (a) compute the multi‑label loss on the logits and
#   (b) optionally render/debug the spatial activations (heatmap overlays).
#
# EXPECTED INPUTS / OUTPUTS
# -----------------------------------------------------------------------------
# Input  (x):  Tensor [B, 1, H, W], grayscale in [-1, +1] from (x-0.5)/0.5 norm,
#              typically H=W=64 (but any square works; we downsample by 4×).
# Outputs (dict of tensors):
#   "logits_given"      : [B, 10]  (use CrossEntropyLoss → softmax at loss)
#   "logits_solution"   : [B, 10]  (use CrossEntropyLoss → softmax at loss)
#   "logits_candidates" : [B, 10]  (use BCEWithLogitsLoss → sigmoid at loss)
#   "cand_maps"         : [B, 10, h, w] (pre‑sigmoid per‑digit spatial logits)
#
# HOW THESE OUTPUTS ARE USED DOWNSTREAM
# -----------------------------------------------------------------------------
# • During training:
#     - Given/Solution: standard CE over 10 classes.
#     - Candidates: BCEWithLogits (multi‑label). Threshold scanning picks a good
#       operating point for precision/recall in non‑empty cells.
# • During inference:
#     - Given/Solution: take argmax of softmax(logits).
#     - Candidates: apply sigmoid(logits_candidates) ≥ threshold (e.g., 0.5) to
#       choose active digits; cand_maps can be visualized/debugged.
#
# CHANGES VS PREVIOUS REVISION
# -----------------------------------------------------------------------------
# • Candidate head is deeper with dilation to increase receptive field without
#   extra downsampling. Stack: 256→128 (3×3), ReLU → 128 (3×3, dil=2), ReLU →
#   10 (1×1) → heatmaps. We keep spatial resolution at 1/4 so maps are sharp.
# • Lower LSE pooling temperature (τ = 0.40) for a crisper max‑like behavior.
# • Continue to expose cand_maps for diagnostics and qualitative checks.
#
# IMPLEMENTATION NOTES
# -----------------------------------------------------------------------------
# • We intentionally avoid BatchNorm to reduce on‑device variance/sync concerns
#   and to keep the parameter count/latency low.
# • All convs use padding to preserve feature map sizes; pooling is only via a
#   single 2×2 MaxPool applied twice (after conv1 and conv2), yielding /4 scale.
# • The trunk feature depth (256) is a sweet spot for quality vs. speed.
#
# =============================================================================
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utility: LogSumExp pooling with temperature (exposed as a function for clarity)
# =============================================================================
def lse_pool_2d(maps: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth global pooling over H×W using log‑sum‑exp.

    Args:
        maps: Tensor of shape [B, C, H, W], pre‑sigmoid spatial logits per class.
        tau : Temperature (τ > 0). As τ→0, behaves like max‑pool; as τ grows,
              becomes more like average.

    Returns:
        pooled: Tensor [B, C] — per‑class pooled logits (still pre‑sigmoid).
    """
    if tau <= 0:
        # Hard global max over spatial locations (non‑smooth)
        return torch.amax(maps, dim=(2, 3))
    # Soft pooling: log ∑ exp(maps/τ) × τ
    return tau * torch.logsumexp(maps / tau, dim=(2, 3))


# =============================================================================
# CellNet — shared trunk + (Given, Solution, Candidates) heads
# =============================================================================
class CellNet(nn.Module):
    """Multi‑head CNN for Sudoku cell understanding.

    Architecture summary:
      Shared trunk (grayscale) → [B,256,h,w]
        • conv1 (1→32, 3×3, pad=1) + ReLU + MaxPool(2)
        • conv2 (32→64, 3×3, pad=1) + ReLU + MaxPool(2)    # now /4 spatial
        • conv3 (64→128, 3×3, pad=1) + ReLU                 # keep resolution
        • conv4 (128→256, 3×3, pad=1) + ReLU

      Heads:
        • Given:    GAP(256) → Linear(256→10) → logits_given
        • Solution: GAP(256) → Linear(256→10) → logits_solution
        • Candidates (spatial):
            256 → 128 (3×3) → ReLU → 128 (3×3, dilation=2) → ReLU → 10 (1×1)
            producing cand_maps [B,10,h,w]; pooled via LogSumExp(τ) to get
            logits_candidates [B,10].

    Notes:
      - We return LOGITS (not probabilities). Losses apply the appropriate
        softmax/sigmoid internally (numerically stable).
      - cand_maps are returned for visualization & diagnostics only.
    """

    # ---------------------------- Initialization ---------------------------- #
    def __init__(self, num_classes: int = 10, lse_tau: float = 0.40):
        super().__init__()
        assert num_classes == 10, "This implementation assumes digits 0..9."

        # --- Hyperparameters kept for clarity/inspection ---
        self.num_classes: int = num_classes
        self.lse_tau: float = float(lse_tau)  # LSE temperature τ

        # -------------------------- Shared conv trunk -------------------------
        # After conv1 + pool:   [B, 32, H/2, W/2]
        # After conv2 + pool:   [B, 64, H/4, W/4]
        # After conv3:          [B,128, H/4, W/4]
        # After conv4:          [B,256, H/4, W/4]
        self.conv1 = nn.Conv2d(in_channels=1,   out_channels=32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)  # 2× downsample

        # Global Average Pool → dense for the two global classification heads
        self.gap      = nn.AdaptiveAvgPool2d((1, 1))  # [B,256,1,1]
        self.feat_dim = 256

        # ----------------------- Given / Solution heads ----------------------
        # Simple, fast, and robust: GAP → Linear(256→10)
        self.head_given    = nn.Linear(self.feat_dim, self.num_classes)
        self.head_solution = nn.Linear(self.feat_dim, self.num_classes)

        # ---------------- Spatial Candidates head (deep + dilated) ----------
        # Keep spatial detail at /4 resolution, enlarge receptive field with
        # a dilated 3×3. The final 1×1 produces 10 per‑digit maps.
        self.cand_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True)
        self.cand_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=True)
        self.cand_conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0, bias=True)
        self.cand_act   = nn.ReLU(inplace=True)

    # ------------------------------- Forward -------------------------------- #
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, 1, H, W] — normalized grayscale cell crops.

        Returns:
            A dict with:
              - "logits_given":      [B,10] class logits for visible GIVEN digit
              - "logits_solution":   [B,10] class logits for SOLVED digit
              - "logits_candidates": [B,10] multi‑label logits from LSE pool
              - "cand_maps":         [B,10,h,w] raw per‑digit spatial logits

        Shapes:
            h = H/4, w = W/4 (integer division from two 2× pools)
        """
        # --- Shared trunk ---
        # Lightweight feature extractor; only two downsamplings (→ /4) to keep
        # candidate maps informative yet cheap to render.
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, H/4, W/4]
        x = F.relu(self.conv3(x))             # [B,128, H/4, W/4]
        x = F.relu(self.conv4(x))             # [B,256, H/4, W/4]

        # --- Global heads: Given/Solution ---
        gap_feat = self.gap(x).view(-1, self.feat_dim)    # [B,256]
        logits_given    = self.head_given(gap_feat)       # [B,10]
        logits_solution = self.head_solution(gap_feat)    # [B,10]

        # --- Spatial candidate head ---
        h = self.cand_act(self.cand_conv1(x))             # [B,128,h,w]
        h = self.cand_act(self.cand_conv2(h))             # [B,128,h,w], dilated RF
        cand_maps = self.cand_conv3(h)                    # [B,10,h,w] (pre‑sigmoid)

        # Pool each per‑digit map to a scalar logit via LogSumExp(τ)
        logits_candidates = lse_pool_2d(cand_maps, tau=self.lse_tau)  # [B,10]

        # Return raw logits (stable for CE/BCEWithLogits) and maps for debug.
        return {
            "logits_given": logits_given,
            "logits_solution": logits_solution,
            "logits_candidates": logits_candidates,
            "cand_maps": cand_maps,
        }


# =============================================================================
# Practical Tips (for readers integrating this module)
# =============================================================================
# • Losses:
#     given/solution → torch.nn.CrossEntropyLoss
#     candidates     → torch.nn.BCEWithLogitsLoss (optionally with focal term)
# • Thresholds:
#     pick candidate threshold via validation scan; typical range 0.45–0.60.
# • Visualization:
#     apply torch.sigmoid(cand_maps[d]) to get a digit‑d heatmap in [0,1].
# • Quantization / Export:
#     trunk & heads are plain ops (Conv/Linear/ReLU/Pooling/LSE). For mobile,
#     ensure your exporter supports logsumexp. If not, τ→0 (max) is a fallback.
# • Input normalization:
#     training code expects grayscale in (x-0.5)/0.5. Keep consistent at runtime.

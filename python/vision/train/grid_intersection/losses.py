import torch
import torch.nn.functional as F

def loss_5map(
    pred, y,
    w_A=1.0, w_H=1.0, w_V=1.0, w_J=2.0,   # ↑ give J a bit more weight early on
    lambda_o=0.5,
    eps=1e-6
):
    """
    pred: (N,6,H,W) where channels = [A,H,V,J,Ox,Oy]
          A,H,V,J are in [0,1]; Ox,Oy are unconstrained logits we will normalize.
    y   : (N,6,H,W) ground truth in the same layout, with A,H,V,J in [0,1] and
          Ox,Oy being unit vectors in [-1,1].
    returns: (loss_total, comps_dict)
    """
    A_p, H_p, V_p, J_p, Ox_p, Oy_p = torch.chunk(pred, 6, dim=1)
    A_y, H_y, V_y, J_y, Ox_y, Oy_y = torch.chunk(y,    6, dim=1)

    # --- Clamp targets to [0,1] (robustness) ---
    A_y = A_y.clamp(0,1); H_y = H_y.clamp(0,1); V_y = V_y.clamp(0,1); J_y = J_y.clamp(0,1)

    # --- BCE losses for masks ---
    bce = F.binary_cross_entropy
    l_A = bce(A_p, A_y)
    l_H = bce(H_p, H_y)
    l_V = bce(V_p, V_y)
    l_J = bce(J_p, J_y)

    # --- Orientation loss (cosine distance, scale-invariant) ---
    O_pred = torch.cat([Ox_p, Oy_p], dim=1)
    O_gt   = torch.cat([Ox_y, Oy_y], dim=1)

    # normalize both to unit vectors
    O_pred = O_pred / (O_pred.norm(dim=1, keepdim=True) + eps)
    O_gt   = O_gt   / (O_gt.norm(dim=1, keepdim=True) + eps)

    cos_sim = (O_pred * O_gt).sum(dim=1, keepdim=True)     # in [-1,1]
    l_O = (1.0 - cos_sim).mean()                           # 0 = perfect

    # --- Weighted sum ---
    loss = (w_A * l_A) + (w_H * l_H) + (w_V * l_V) + (w_J * l_J) + (lambda_o * l_O)

    comps = {
        "bce_A": l_A.detach(),
        "bce_H": l_H.detach(),
        "bce_V": l_V.detach(),
        "bce_J": l_J.detach(),
        "cos_O": l_O.detach(),
        "w_A": torch.tensor(w_A),
        "w_H": torch.tensor(w_H),
        "w_V": torch.tensor(w_V),
        "w_J": torch.tensor(w_J),
        "lambda_o": torch.tensor(lambda_o),
    }
    return loss, comps
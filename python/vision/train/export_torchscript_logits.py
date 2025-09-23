# vision/train/export_torchscript_logits.py
import argparse, sys, os, torch
import torch.nn as nn

# Ensure repo root on sys.path when run with -m
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from vision.models.cnn_small import CNN28  # uses your repo's CNN28


def load_core(model_path: str, device: str = "cpu") -> nn.Module:
    """Build CNN28 and load weights from a variety of checkpoint formats."""
    state = torch.load(model_path, map_location=device)

    # Common formats:
    #  1) torch.save({'model': state_dict, ...})
    #  2) torch.save(state_dict)
    if isinstance(state, dict) and "state_dict" in state and "model" not in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Strip 'module.' if the checkpoint came from DataParallel
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    core = CNN28(num_classes=10)  # no in_ch in your repo's CNN28
    missing, unexpected = core.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[export] NOTE: load_state_dict mismatch: missing={missing}, unexpected={unexpected}")

    core.eval()
    core.to(device)
    return core


class LogitsWrapper(nn.Module):
    """
    Wraps the core model. If core.forward returns probabilities (rows ~sum to 1 and values in [0,1]),
    we convert to logits via log(p + eps) so downstream temperature scaling works as intended.
    Otherwise, we just return the core output (already logits).
    """
    def __init__(self, core: nn.Module, prob_like: bool):
        super().__init__()
        self.core = core
        self.prob_like = prob_like

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        if self.prob_like:
            eps = 1e-8
            y = torch.log(torch.clamp(y, eps, 1.0 - eps))
        return y


def looks_like_probs(y: torch.Tensor) -> bool:
    """
    Heuristic: returns True if outputs look like probabilities.
    Conditions (on a small batch):
      - all values in [~0, ~1]
      - each row sums ~1
    """
    if y.ndim != 2 or y.size(1) != 10:
        return False
    with torch.no_grad():
        ymin = float(y.min().item())
        ymax = float(y.max().item())
        row_sums = y.sum(dim=1)
        near_01 = (ymin >= -1e-6) and (ymax <= 1.0 + 1e-6)
        near_1 = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=0)
    return bool(near_01 and near_1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to checkpoint (best.pt)")
    ap.add_argument("--out", required=True, help="Output TorchScript path (.ptl)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--force-probs", action="store_true",
                    help="Force treating core outputs as probabilities (convert to logits).")
    ap.add_argument("--force-logits", action="store_true",
                    help="Force treating core outputs as logits (no conversion).")
    args = ap.parse_args()

    core = load_core(args.model, device=args.device)

    # Probe on dummy data to auto-detect prob/logits if not forced
    with torch.no_grad():
        dummy = torch.randn(4, 1, 28, 28, device=args.device)
        probe = core(dummy)

    if args.force_probs and args.force_logits:
        print("[export] --force-probs and --force-logits are mutually exclusive.", file=sys.stderr)
        sys.exit(2)

    if args.force_probs:
        prob_like = True
    elif args.force_logits:
        prob_like = False
    else:
        prob_like = looks_like_probs(probe)

    print(f"[export] Detection: treating core outputs as "
          f"{'PROBABILITIES' if prob_like else 'LOGITS'}")

    wrapper = LogitsWrapper(core, prob_like=prob_like).to(args.device).eval()

    # Script & save
    scripted = torch.jit.script(wrapper)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scripted.save(args.out)
    print(f"[export] Saved TorchScript (logits) to: {args.out}")


if __name__ == "__main__":
    main()
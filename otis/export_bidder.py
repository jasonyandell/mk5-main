"""otis/export_bidder.py — lift trunk+pricing into a vanilla MarginNet checkpoint.

Reads an OtisNet checkpoint (either arm), copies the trunk+pricing weights into a
``{"model_state": ..., "feature_dim": ...}`` checkpoint that
``champion.value_bidder.load_margin_net`` loads STRICT, and verifies the export is
BIT-EXACT against the source OtisNet's pricing path on 100 rows.

For treatment this proves the pricing head the ValueBidder consumes is untouched
by the auxiliary organs at serving time; for control it proves the arm is exactly
the jud-v0 MarginNet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from champion.margin_net import FEATURE_DIM, MarginNet
from champion.value_bidder import load_margin_net
from otis.model import OtisNet

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_otis(ckpt_path: str | Path, device: str = "cpu") -> tuple[OtisNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = OtisNet(in_dim=ckpt.get("feature_dim", FEATURE_DIM),
                    treatment=bool(ckpt.get("treatment", False))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export(ckpt_path: str | Path, out_path: str | Path, *,
           device: str = "cpu", n_verify: int = 100) -> dict:
    """Export trunk+pricing → MarginNet checkpoint at ``out_path``; verify.

    Returns a dict with ``strict_load_ok`` and ``max_abs_diff`` (must be 0.0 for
    bit-exactness).
    """
    otis, ckpt = load_otis(ckpt_path, device)
    state = otis.export_margin_state_dict()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": state, "feature_dim": ckpt.get("feature_dim", FEATURE_DIM)},
               out_path)

    # STRICT reload through the exact consumer path.
    margin = load_margin_net(out_path, device=device)  # strict load_state_dict
    strict_ok = isinstance(margin, MarginNet)

    # Bit-exact check on n_verify random rows.
    torch.manual_seed(0)
    x = torch.randn(n_verify, ckpt.get("feature_dim", FEATURE_DIM), device=device)
    with torch.no_grad():
        a = otis.pricing_logits(x)   # OtisNet pricing path
        b = margin(x)                # exported MarginNet
    max_abs = float((a - b).abs().max().item())
    bit_exact = torch.equal(a, b)
    result = {
        "src_ckpt": str(ckpt_path),
        "out": str(out_path),
        "arm": ckpt.get("arm", "?"),
        "strict_load_ok": bool(strict_ok),
        "n_verify": n_verify,
        "max_abs_diff": max_abs,
        "bit_exact": bool(bit_exact),
    }
    print(f"[export] {ckpt.get('arm','?')} → {out_path}  "
          f"strict_load={strict_ok}  bit_exact={bit_exact}  "
          f"max_abs_diff={max_abs:.3e}", flush=True)
    if not (strict_ok and bit_exact):
        raise AssertionError(f"export verification FAILED: {result}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="export otis trunk+pricing → MarginNet ckpt")
    ap.add_argument("--ckpt", required=True, help="OtisNet checkpoint (from otis.train)")
    ap.add_argument("--out", type=Path, required=True,
                    help="e.g. otis/models/otis_v0_control.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-verify", type=int, default=100)
    args = ap.parse_args()
    export(args.ckpt, args.out, device=args.device, n_verify=args.n_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

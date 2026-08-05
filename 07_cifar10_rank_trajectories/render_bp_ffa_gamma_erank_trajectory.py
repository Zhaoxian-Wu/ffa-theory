"""Render the completed BP-versus-FFA Gamma-rank trajectory without retraining."""
from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))

from train_bp_ffa_gamma_erank_trajectory import plot_trajectory  # noqa: E402


def main() -> None:
    results_dir = ROOT / "results" / "bp_ffa_gamma_erank_trajectory"
    payload_path = results_dir / "trajectory_completed.json"
    with payload_path.open() as handle:
        payload = json.load(handle)
    plot_trajectory(payload, results_dir / "bp_ffa_gamma_erank_trajectory")


if __name__ == "__main__":
    main()

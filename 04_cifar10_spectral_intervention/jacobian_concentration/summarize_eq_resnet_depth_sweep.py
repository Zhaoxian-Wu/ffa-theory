"""Aggregate the exact eq:resnet_arch depth sweep into one JSON table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (4, 8, 16, 32)
SOURCES = {
    "zero_noise": {
        4: ROOT / "results" / "eq_resnet_arch_tau_j_L4_zero_noise.json",
        8: ROOT / "results" / "eq_resnet_arch_tau_j_L8_zero_noise.json",
        16: ROOT / "results" / "eq_resnet_arch_tau_j_zero_noise.json",
        32: ROOT / "results" / "eq_resnet_arch_tau_j_L32_zero_noise.json",
    },
    "orthogonal_noise_1e-3": {
        4: ROOT / "results" / "eq_resnet_arch_tau_j_L4_perturbed.json",
        8: ROOT / "results" / "eq_resnet_arch_tau_j_L8_perturbed.json",
        16: ROOT / "results" / "eq_resnet_arch_tau_j.json",
        32: ROOT / "results" / "eq_resnet_arch_tau_j_L32_perturbed.json",
    },
}


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text())
    layers = payload["layers"]
    full_tail = layers[0]
    return {
        "depth": payload["depth"],
        "test_accuracy": payload["training"]["test_accuracy"],
        "Gamma_L_effective_rank": payload["Gamma_L"]["effective_rank"],
        "full_tail": {
            key: full_tail[key]
            for key in (
                "tau_J_sup",
                "tau_J_L2",
                "s_plus_star",
                "s_minus_star",
                "rho_sup",
                "rho_L2",
            )
        },
        "all_nonempty_tails": {
            "rho_sup_min": min(layer["rho_sup"] for layer in layers),
            "rho_sup_max": max(layer["rho_sup"] for layer in layers),
            "rho_L2_min": min(layer["rho_L2"] for layer in layers),
            "rho_L2_max": max(layer["rho_L2"] for layer in layers),
        },
        "source": str(path.relative_to(ROOT)),
    }


def main() -> None:
    result = {
        "experiment": "exact_eq_resnet_arch_depth_sweep",
        "scope": (
            "Controlled symmetry-protected construction only. Every block is exactly "
            "h + ReLU(LayerNorm(W h)) / L with W 1 = 0; it is not a generic ResNet result."
        ),
        "conditions": {},
    }
    for condition, paths in SOURCES.items():
        entries = [summarize(paths[depth]) for depth in DEPTHS]
        if [entry["depth"] for entry in entries] != list(DEPTHS):
            raise RuntimeError(f"Depth metadata does not match {DEPTHS} for {condition}.")
        result["conditions"][condition] = entries
    output = ROOT / "results" / "eq_resnet_arch_depth_sweep.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()

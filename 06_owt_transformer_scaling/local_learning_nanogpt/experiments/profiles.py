"""Training profile registry and schedule helpers."""

from __future__ import annotations

from local_learning_nanogpt.experiments.specs import ProfileSpec, ScaleSpec, ScalingPreset


TOKENS_PER_STEP = 32 * 256


PROFILE_REGISTRY: dict[str, ProfileSpec] = {
    "quick": ProfileSpec("quick", "Short quick-scan training runs"),
    "chinchilla": ProfileSpec("chinchilla", "Token-budget scaling profile"),
}


def list_profiles() -> list[str]:
    return list(PROFILE_REGISTRY)


def get_profile_spec(name: str) -> ProfileSpec:
    if name not in PROFILE_REGISTRY:
        raise KeyError(f"Unknown profile: {name}")
    return PROFILE_REGISTRY[name]


def resolve_token_multiplier(profile_name: str, override: float | None) -> float | None:
    if profile_name != "chinchilla":
        return None
    return override if override is not None else 20.0


def resolve_max_iters(
    *,
    profile_name: str,
    dataset_name: str,
    scale: ScaleSpec,
    max_iters_override: int | None,
    token_multiplier_override: float | None,
    preset: ScalingPreset | None,
) -> int:
    if max_iters_override is not None:
        return max_iters_override
    if preset and preset.default_max_iters is not None:
        return preset.default_max_iters
    if profile_name == "chinchilla":
        token_multiplier = resolve_token_multiplier(profile_name, token_multiplier_override)
        if scale.approx_params is None or token_multiplier is None:
            raise ValueError("Chinchilla profile requires approx_params on every selected scale.")
        return int(token_multiplier * scale.approx_params / TOKENS_PER_STEP)
    if dataset_name == "shakespeare":
        return 3000
    if dataset_name == "owt_small" and scale.name == "xlarge":
        return 10000
    return 5000


def resolve_probe_iters(
    *,
    probe_iters_override: int | None,
    preset: ScalingPreset | None,
) -> int:
    if probe_iters_override is not None:
        return probe_iters_override
    if preset and preset.default_probe_iters is not None:
        return preset.default_probe_iters
    return 0


def resolve_warmup_iters(
    *,
    profile_name: str,
    dataset_name: str,
    scale_name: str,
    max_iters: int,
    preset: ScalingPreset | None,
) -> int:
    if profile_name == "quick":
        if dataset_name == "shakespeare":
            return 100
        if scale_name == "xlarge":
            return 400
        return 200
    if preset and preset.output_filename == "scaling_law_fair_xlarge.json":
        return min(500, max_iters // 40)
    return min(400, max_iters // 20)


def resolve_eval_interval(
    *,
    profile_name: str,
    max_iters: int,
    eval_interval_override: int | None,
    preset: ScalingPreset | None,
) -> int:
    if eval_interval_override is not None:
        return eval_interval_override
    if profile_name == "quick":
        return 500
    if preset and preset.output_filename == "scaling_law_fair_xlarge.json":
        return max(200, max_iters // 200)
    return max(100, int(max_iters * 0.05))


def resolve_eval_batches(*, profile_name: str, dataset_name: str) -> int:
    if profile_name == "chinchilla":
        return 10
    if dataset_name == "shakespeare":
        return 1
    return 5


def resolve_erank_interval(*, profile_name: str, eval_interval: int) -> int:
    if profile_name == "chinchilla":
        return eval_interval * 4
    return eval_interval


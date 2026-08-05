"""Core model and optimizer implementations."""

from local_learning_nanogpt.core.ffa_gpt import FFAGPT, FFAGPTConfig
from local_learning_nanogpt.core.muon_optimizer import Muon, make_muon_optimizer

__all__ = ["FFAGPT", "FFAGPTConfig", "Muon", "make_muon_optimizer"]


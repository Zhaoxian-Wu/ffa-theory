"""The nine methods compared in the ResNet experiments."""
from .bp import BPTrainer
from .local_supervised import NoklandTrainer, SFFTrainer, DistanceForwardTrainer
from .vanilla_ffa import LabelEnumTrainer
from .scff import SCFFTrainer
from .symba import SymBaTrainer
from .layer_collab import LayerCollabTrainer
from .trifecta import TrifectaTrainer

TRAINERS = {
    "bp": BPTrainer,
    "nokland_lpredsim": NoklandTrainer,
    "sff": SFFTrainer,
    "distance_forward": DistanceForwardTrainer,
    "vanilla_ffa": LabelEnumTrainer,
    "scff": SCFFTrainer,
    "symba": SymBaTrainer,
    "layer_collab": LayerCollabTrainer,
    "trifecta": TrifectaTrainer,
}


def build_trainer(algorithm, config):
    trainer = TRAINERS[algorithm](config)
    trainer.algo = algorithm
    return trainer

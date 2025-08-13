#parameter_class.py
# Provides dataclass for setting model parameters, as well
# as irace optimized hyperparams (on vpn data) to easily access and swap
from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class ModelParameters:
    model_name: str

    learning_rate: float
    scheduler_gamma: float
    scheduler_milestones: List[int]
    weight_decay: float
    momentum: float

    training_epochs: int = 1
    episodes_per_epoch: int = 20000
    validation_runs: int = 100
    n_way: int = 5
    n_classes: int = 5
    n_support: int = 5
    n_query: int = 30
    n_epochs: int = training_epochs

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

class HyperparamStore:
    def __init__(self):
        self.models = {
            "hidden": {
                "bdcspn": ModelParameters(
                    model_name="bdcspn",
                    learning_rate=0.000970,
                    scheduler_gamma=0.423808,
                    momentum=0.947331,
                    scheduler_milestones=[20,50,100],
                    weight_decay=0,
                ),
                "bdlpn": ModelParameters(
                    model_name="bdlpn",
                    learning_rate=0.000970,
                    scheduler_gamma=0.423808,
                    momentum=0.947331,
                    scheduler_milestones=[20,50,100],
                    weight_decay=0,
                ),
                "relation_networks": ModelParameters(
                    model_name="relation_networks",
                    learning_rate=0.000699,
                    scheduler_gamma=0.146967,
                    momentum=0.931401,
                    scheduler_milestones=[20,50,100],
                    weight_decay=0,
                    training_epochs=1,
                    episodes_per_epoch=20000
                ),
                "laplacian_shot": ModelParameters(
                    model_name="laplacian_shot",
                    learning_rate=0.000585,
                    scheduler_gamma=0.297849,
                    momentum=0.948601,
                    scheduler_milestones=[20,50,100],
                    weight_decay=0,
                ),
                "ptmap": ModelParameters(
                    model_name="ptmap",
                    learning_rate=0.001,
                    scheduler_gamma=0.31544,
                    momentum=0.647609,
                    scheduler_milestones=[2, 5, 10],
                    weight_decay=0,
                    training_epochs=100,
                    episodes_per_epoch=20000,
                ),

            },
            "linear": {
                "bdcspn": ModelParameters(
                    model_name="bdcspn",
                    learning_rate=0.009344,
                    scheduler_gamma=0.13455,
                    momentum=0.78392,
                    scheduler_milestones=[20,50,100],
                    weight_decay=0,
                ),
                "laplacian_shot": ModelParameters(
                    model_name="laplacian_shot",
                    learning_rate=0.006573,
                    scheduler_gamma=0.26211,
                    momentum=0.358739,
                    scheduler_milestones=[20, 50, 100],
                    weight_decay=0.000001,
                )
            }
        }

    def get_model_params(self, model_name: str, model_type: str) -> Optional[ModelParameters]:
        return self.models.get(model_type, {}).get(model_name, None)



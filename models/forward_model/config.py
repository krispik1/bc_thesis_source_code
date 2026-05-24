from dataclasses import field, dataclass
from typing import Dict, List

import torch

from models.config import ModelTrainConfig


@dataclass
class ForwardModelTrainConfig(ModelTrainConfig):
    model_name: str = "forward"

    num_epochs: int = 25
    early_stopping_patience: int = 7
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32

    # Architecture parameters
    input_dimension: int = 37
    shared_hidden_dimension: int = 512
    n_shared_hidden_layers: int = 1
    head_hidden_dimension: int = 128
    output_layers: Dict[str, int] = field(default_factory=lambda: {
        "configuration": 7,
        "end_effector_position": 3,
        "end_effector_rotation": 4,
        "magnet": 1,
        "goal_obj6D_position": 3,
        "goal_obj6D_rotation": 4,
        "obstacle6D_position": 3,
        "obstacle6D_rotation": 4
    })

    # Head weights
    lambdas: Dict[str, int] = field(default_factory=lambda: {
        "configuration": 1.0,
        "end_effector_position": 1.0,
        "end_effector_rotation": 1.0,
        "magnet": 1.0,
        "goal_obj6D_position": 1.0,
        "goal_obj6D_rotation": 1.0,
        "obstacle6D_position": 1.0,
        "obstacle6D_rotation": 1.0
    })

    # Dropout during training
    dropout_rate: float = 0.0

    # Plateau scheduler parameters
    scheduler_factor: float = 0.3
    scheduler_patience: int = 8

    # Cosine scheduler parameters
    T_max: int = 0
    eta_min: float = 1e-6

    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "loss_total",
        "loss_configuration",
        "loss_end_effector",
        "loss_magnet",
        "loss_goal_obj6D",
        "loss_obstacle6D",

        "configuration_mae",
        "end_effector_position_mae",
        "end_effector_rotation_mae",
        "magnet_acc",
        "goal_obj6D_position_mae",
        "goal_obj6D_rotation_mae",
        "obstacle6D_position_mae",
        "obstacle6D_rotation_mae"
    ])

    # Metrics symbol dict
    metrics_symbols: Dict[str, str] = field(default_factory=lambda: {
        "configuration_mae": r'$\mathbf{\theta}$ MAE[rad]',
        "end_effector_position_mae" : r'$\mathbf{ef}_{xyz}$ MAE[m]',
        "end_effector_rotation_mae": r'$\mathbf{ef}_{R}$ MGE[rad]',
        "magnet_acc" : r'$mgt$ acc',
        "goal_obj6D_position_mae" : r'$\mathbf{g}_{xyz}$ MAE[m]',
        "goal_obj6D_rotation_mae" : r'$\mathbf{g}_{R}$ MGE[rad]',
        "obstacle6D_position_mae" : r'$\mathbf{o}_{xyz}$ MAE[m]',
        "obstacle6D_rotation_mae" : r'$\mathbf{o}_{R}$ MGE[rad]',
    })

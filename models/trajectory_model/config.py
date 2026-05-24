from dataclasses import dataclass, field
from typing import Dict, List

import torch

from remake.models.config import ModelTrainConfig


@dataclass
class TrajectoryModelTrainConfig(ModelTrainConfig):
    model_name: str = "trajectory"

    num_epochs: int = 25
    early_stopping_patience: int = 8
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 64

    # Gradient clipping
    gradient_clip: float = 2.0

    # Architecture parameters
    input_dimension: int = 58
    n_timesteps: int = 50
    n_gru: int = 2
    d_gru: int = 768
    hidden_head_dimension: int = 256
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

    # Weights
    lambdas: Dict[str, float] = field(default_factory=lambda: {
        "acceleration": 0.0,
        "step": 0.0,
        "length": 0.0,
        "angle": 0.0,
    })

    # Plateau scheduler parameters
    scheduler_factor: float = 1.0
    scheduler_patience: int = 8

    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "loss_total",
        "loss_trajectory",
        "loss_initial_position",
        "loss_goal_position",
        "distance_to_initial_position",
        "distance_to_goal_position",
        "average_spacing_between_points",
        "max_spacing_deviation",
        "average_angle_between_points",
        "minimum_angle_between_points",

        "configuration_rectification_mae",
        "end_effector_position_rectification_mae",
        "end_effector_rotation_rectification_mae",
        "magnet_rectification_mae",
        "goal_obj6D_position_rectification_mae",
        "goal_obj6D_rotation_rectification_mae",
        "obstacle6D_position_rectification_mae",
        "obstacle6D_rotation_rectification_mae",

        "configuration_rectification_l2",
        "end_effector_position_rectification_l2",
        "end_effector_rotation_rectification_l2",
        "magnet_rectification_l2",
        "goal_obj6D_position_rectification_l2",
        "goal_obj6D_rotation_rectification_l2",
        "obstacle6D_position_rectification_l2",
        "obstacle6D_rotation_rectification_l2",

        "tail_mean_step_size",
        "tail_fraction_static",
        "estimated_effective_length",
    ])

    # Metrics symbol dict
    metrics_symbols: Dict[str, str] = field(default_factory=lambda: {
        "distance_to_initial_position": r"${d}_{init}[m]$",
        "distance_to_goal_position": r"${d}_{goal}[m]$",
        "average_spacing_between_points": r"$\bar{s}[m]$",
        "average_angle_between_points": r"$\bar{\theta}[^{\circ}]$",

        "configuration_rectification_mae": r'$\mathbf{\theta}$ MAE[rad]',
        "end_effector_position_rectification_mae": r'$\mathbf{ef}_{xyz}$ MAE[m]',
        "end_effector_rotation_rectification_mae": r'$\mathbf{ef}_{R}$ MGE[rad]',
        "magnet_rectification_mae": r'$mgt$ MAE',
        "goal_obj6D_position_rectification_mae": r'$\mathbf{g}_{xyz}$ MAE[m]',
        "goal_obj6D_rotation_rectification_mae": r'$\mathbf{g}_{R}$ MGE[rad]',
        "obstacle6D_position_rectification_mae": r'$\mathbf{o}_{xyz}$ MAE[m]',
        "obstacle6D_rotation_rectification_mae": r'$\mathbf{o}_{R}$ MGE[rad]',

        "tail_fraction_static" : r'f_{tail}',
        "estimated_effective_length": r'L_{eff}',
    })
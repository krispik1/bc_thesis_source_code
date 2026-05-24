from dataclasses import dataclass, field
from typing import List, Dict

import torch

from models.config import ModelTrainConfig


@dataclass
class InverseModelTrainConfig(ModelTrainConfig):
    model_name: str = "inverse_model"

    num_epochs: int = 25
    early_stopping_patience: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64

    # Architecture parameters
    input_dimension: int = 50
    hidden_dimension: int = 512
    output_dimension: int = 8
    n_hidden_layers: int = 6

    # Plateau scheduler parameters
    scheduler_factor: float = 1.0
    scheduler_patience: int = 8

    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "loss_total",
        "loss_delta_q",
        "loss_delta_magnet",

        "delta_q_mae",
        "delta_magnet_mae",
        "delta_magnet_acc"
    ])

    # Metrics symbol dict
    metrics_symbols: Dict[str, str] = field(default_factory=lambda: {
        "delta_q_mae": r'$\Delta \mathbf{\theta}$ MAE[rad]',
        "delta_magnet_mae": r'$\Delta mgt$ MAE',
    })

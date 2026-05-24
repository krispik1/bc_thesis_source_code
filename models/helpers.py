import csv
import logging
from pathlib import Path
from typing import Dict, Set, Any

import torch
from torch import nn

from remake.models.config import ModelTrainConfig


def setup_logger(
        log_dir: str,
        model: str
) -> logging.Logger:
    """
    Sets up logger using library logging.
    """
    logger = logging.getLogger(f"{model}")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(str(Path(log_dir) / f"{model}.log"))
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def load_existing_keys(
        file_path: str
) -> Set[Any]:
    """
    Loads existing keys from a csv file. Keys are then checked to avoid duplicate writes.

    :param file_path: File path to csv file.
    :return: Set of keys from the file.
    """
    keys = set()
    try:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add((int(row["run_id"]), int(row["epoch_id"])))
    except FileNotFoundError:
        pass
    return keys

def move_batch_to_device(
        batch: Dict[str, torch.Tensor],
        device: str
) -> Dict[str, torch.Tensor]:
    """
    Moves batch to device.

    :param batch: Batch of data.
    :param device: Device to move batch to.
    :return: Moved batch to device.
    """
    return {key: value.to(device) for key, value in batch.items()}

def create_optimizer(
        model: nn.Module,
        cfg: ModelTrainConfig,
) -> torch.optim.Optimizer:
    """
    Returns optimizer for given model and configuration.

    :param model: Model.
    :param cfg: Training configuration.
    :return: Optimizer.
    """

    match cfg.optimizer:
        case "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        case "AdamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        case "RMSprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        case "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=cfg.learning_rate,
                momentum=0.9
            )
        case _:
            raise ValueError(f"Unexpected optimizer type {cfg.optimizer}.")
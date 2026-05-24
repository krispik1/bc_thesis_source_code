import csv
import os
from logging import Logger
from pathlib import Path
from typing import Dict, List, Any

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from models.forward_model.config import ForwardModelTrainConfig
from models.forward_model.loss_function import quaternion_geodesic_loss, forward_model_loss
from models.helpers import move_batch_to_device, create_optimizer, load_existing_keys


@torch.no_grad()
def compute_forward_metrics(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes evaluation metrics for each head.

    :param prediction: Predicted state vector.
    :param target: Target state vector.
    :return: Dict of metrics.
    """

    metrics: Dict[str, float] = {}

    configuration_mae = torch.mean(
        torch.abs(prediction["configuration"] - target["configuration"])
    )

    end_effector_position_mae = torch.mean(
        torch.abs(prediction["end_effector_position"] - target["end_effector"][:, :3])
    )
    end_effector_rotation_mae = quaternion_geodesic_loss(
        prediction["end_effector_rotation"], target["end_effector"][:, 3:]
    )

    magnet_probability = torch.sigmoid(prediction["magnet"])
    magnet_prediction = (magnet_probability >= 0.5).float()
    magnet_acc = (
        magnet_prediction == target["magnet"].view_as(prediction["magnet"])
    ).float().mean()

    goal_obj6D_position_mae = torch.mean(
        torch.abs(prediction["goal_obj6D_position"] - target["goal_obj6D"][:, :3])
    )
    goal_obj6D_rotation_mae = quaternion_geodesic_loss(
        prediction["goal_obj6D_rotation"], target["goal_obj6D"][:, 3:]
    )

    obstacle6D_position_mae = torch.mean(
        torch.abs(prediction["obstacle6D_position"] - target["obstacle6D"][:, :3])
    )
    obstacle6D_rotation_mae = quaternion_geodesic_loss(
        prediction["obstacle6D_rotation"], target["obstacle6D"][:, 3:]
    )

    metrics["configuration_mae"] = configuration_mae.item()
    metrics["end_effector_position_mae"] = end_effector_position_mae.item()
    metrics["end_effector_rotation_mae"] = end_effector_rotation_mae.item()
    metrics["magnet_acc"] = magnet_acc.item()
    metrics["goal_obj6D_position_mae"] = goal_obj6D_position_mae.item()
    metrics["goal_obj6D_rotation_mae"] = goal_obj6D_rotation_mae.item()
    metrics["obstacle6D_position_mae"] = obstacle6D_position_mae.item()
    metrics["obstacle6D_rotation_mae"] = obstacle6D_rotation_mae.item()

    return metrics

def forward_train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        cfg: ForwardModelTrainConfig
) -> Dict[str, float]:
    """
    Trains one epoch and keeps track of loss values and evaluation metrics. Model's parameters change.

    :param model: Trained model.
    :param dataloader: Dataloader.
    :param optimizer: Optimiser.
    :param device: Device on which the epoch will run.
    :param cfg: Configuration of the model.
    :return: Epoch training metrics.
    """
    model.train()

    running = {
        "loss_total": 0.0,
        "loss_configuration": 0.0,
        "loss_end_effector": 0.0,
        "loss_magnet": 0.0,
        "loss_goal_obj6D": 0.0,
        "loss_obstacle6D": 0.0,
        "configuration_mae": 0.0,
        "end_effector_position_mae": 0.0,
        "end_effector_rotation_mae": 0.0,
        "magnet_acc": 0.0,
        "goal_obj6D_position_mae": 0.0,
        "goal_obj6D_rotation_mae": 0.0,
        "obstacle6D_position_mae": 0.0,
        "obstacle6D_rotation_mae": 0.0,
    }

    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        target = {
            "configuration": batch["next_configuration"],
            "end_effector": batch["next_end_effector"],
            "magnet": batch["next_magnet"],
            "goal_obj6D": batch["next_goal_obj6D"],
            "obstacle6D": batch["next_obstacle6D"],
        }

        optimizer.zero_grad()

        prediction = model(x)
        loss, loss_stats = forward_model_loss(prediction, target, cfg.lambdas)

        loss.backward()

        optimizer.step()

        metric_stats = compute_forward_metrics(prediction, target)

        for key, value in loss_stats.items():
            running[key] += value
        for key, value in metric_stats.items():
            running[key] += value

        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in running.items()}


@torch.no_grad()
def forward_validate_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        cfg: ForwardModelTrainConfig
) -> Dict[str, float]:
    """
    Validates the model for one epoch and keeps track of loss values and evaluation metrics.

    :param model: Evaluated model.
    :param dataloader: Dataloader.
    :param device: Device on which the epoch will run.
    :param cfg: Configuration of the model.
    :return: Epoch validation metrics.
    """

    model.eval()

    running = {
        "loss_total": 0.0,
        "loss_configuration": 0.0,
        "loss_end_effector": 0.0,
        "loss_magnet": 0.0,
        "loss_goal_obj6D": 0.0,
        "loss_obstacle6D": 0.0,
        "configuration_mae": 0.0,
        "end_effector_position_mae": 0.0,
        "end_effector_rotation_mae": 0.0,
        "magnet_acc": 0.0,
        "goal_obj6D_position_mae": 0.0,
        "goal_obj6D_rotation_mae": 0.0,
        "obstacle6D_position_mae": 0.0,
        "obstacle6D_rotation_mae": 0.0,
    }

    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        target = {
            "configuration": batch["next_configuration"],
            "end_effector": batch["next_end_effector"],
            "magnet": batch["next_magnet"],
            "goal_obj6D": batch["next_goal_obj6D"],
            "obstacle6D": batch["next_obstacle6D"],
        }

        prediction = model(x)
        loss, loss_stats = forward_model_loss(prediction, target, cfg.lambdas)
        metric_stats = compute_forward_metrics(prediction, target)

        for key, value in loss_stats.items():
            running[key] += value
        for key, value in metric_stats.items():
            running[key] += value

        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in running.items()}

def plot_metrics(
        cfg: ForwardModelTrainConfig,
        history: Dict[str, List[float]],
        run_id: int,
        plots_dir: str,
):
    """
    Visualises evaluation metrics for both training and validation.

    :param cfg: Configuration of the model.
    :param history: History of the run.
    :param run_id: Run ID.
    :param plots_dir: Path to the plots/graphs directory.
    """
    fig = plt.figure(figsize=(10, 8))

    n_points = len(history["val_loss_total"])
    x_axis = range(3, n_points + 1)

    plt.subplot(3, 1, 1)
    plt.plot(x_axis, history["val_loss_total"][2:], marker="", label="val_loss")
    plt.plot(x_axis, history["train_loss_total"][2:], marker="", label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"run={run_id} lr={cfg.learning_rate} shared={cfg.shared_hidden_dimension} head={cfg.head_hidden_dimension} layers={cfg.n_shared_hidden_layers}")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x_axis, history["val_configuration_mae"][2:], marker="", label="configuration_mae")
    plt.plot(x_axis, history["val_end_effector_position_mae"][2:], marker="", label="ee_position_mae")
    plt.plot(x_axis, history["val_end_effector_rotation_mae"][2:], marker="", label="ee_rotation_mae")
    plt.plot(x_axis, history["val_goal_obj6D_position_mae"][2:], marker="", label="goal_position_mae")
    plt.plot(x_axis, history["val_goal_obj6D_rotation_mae"][2:], marker="", label="goal_rotation_mae")
    plt.plot(x_axis, history["val_obstacle6D_position_mae"][2:], marker="", label="obstacle_position_mae")
    plt.plot(x_axis, history["val_obstacle6D_rotation_mae"][2:], marker="", label="obstacle_rotation_mae")
    plt.xlabel("epoch")
    plt.ylabel("val metrics")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x_axis, history["train_configuration_mae"][2:], marker="", label="configuration_mae")
    plt.plot(x_axis, history["train_end_effector_position_mae"][2:], marker="", label="ee_position_mae")
    plt.plot(x_axis, history["train_end_effector_rotation_mae"][2:], marker="", label="ee_rotation_mae")
    plt.plot(x_axis, history["train_goal_obj6D_position_mae"][2:], marker="", label="goal_position_mae")
    plt.plot(x_axis, history["train_goal_obj6D_rotation_mae"][2:], marker="", label="goal_rotation_mae")
    plt.plot(x_axis, history["train_obstacle6D_position_mae"][2:], marker="", label="obstacle_position_mae")
    plt.plot(x_axis, history["train_obstacle6D_rotation_mae"][2:], marker="", label="obstacle_rotation_mae")
    plt.xlabel("epoch")
    plt.ylabel("train metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(str(Path(plots_dir) / f"run_{run_id:03d}.png"), bbox_inches="tight")
    plt.close(fig)

def fit_forward_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ForwardModelTrainConfig,
        logger: Logger,
        plots_dir: str,
        results_path: str,
        checkpoint_dir: str = "best_forward_model.pt",
        run_id: int = 1,
) -> Dict[str, list]:
    """
    Trains and then validates the model.

    :param model: Model.
    :param train_loader: Dataloader with training data.
    :param val_loader: Dataloader with validation data.
    :param cfg: Configuration of the model.
    :param logger: Logger.
    :param plots_dir: Path to the plots directory.
    :param results_path: Path to the results file.
    :param checkpoint_dir: Path to the checkpoint directory.
    :param run_id: Run ID.
    :return: History of the run.
    """
    model = model.to(cfg.device)

    optimizer = create_optimizer(model, cfg)

    # If scheduler used, initialize it
    scheduler = None
    if cfg.scheduler_factor != 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=1e-6,
        )

    history = {
        "train_loss_total": [],
        "val_loss_total": [],
        "train_configuration_mae": [],
        "val_configuration_mae": [],
        "train_end_effector_position_mae": [],
        "val_end_effector_position_mae": [],
        "train_end_effector_rotation_mae": [],
        "val_end_effector_rotation_mae": [],
        "train_magnet_acc": [],
        "val_magnet_acc": [],
        "train_goal_obj6D_position_mae": [],
        "val_goal_obj6D_position_mae": [],
        "train_goal_obj6D_rotation_mae": [],
        "val_goal_obj6D_rotation_mae": [],
        "train_obstacle6D_position_mae": [],
        "val_obstacle6D_position_mae": [],
        "train_obstacle6D_rotation_mae": [],
        "val_obstacle6D_rotation_mae": [],
    }

    best_val_loss = float("inf")
    start_epoch = 1

    # Load from checkpoint if exists
    path = Path(checkpoint_dir) / f"run_{run_id:03d}.pt"
    if path.exists():
        optimizer.load_state_dict(torch.load(path)["optimizer_state_dict"])
        history = torch.load(path)["history"]
        best_val_loss = torch.load(path)["best_val_loss"]
        start_epoch = torch.load(path)["epoch"] + 1
        scheduler_dict: Dict[str, Any] = torch.load(path).get("scheduler_state_dict", None)
        if scheduler_dict is not None and scheduler is not None:
            scheduler.load_state_dict(scheduler_dict)

    plot_metrics(cfg, history, run_id, plots_dir)
    existing_keys = load_existing_keys(results_path)

    for epoch in range(start_epoch, cfg.num_epochs + 1):

        train_stats = forward_train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=cfg.device,
            cfg=cfg
        )

        val_stats = forward_validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=cfg.device,
            cfg=cfg
        )

        if scheduler is not None and cfg.scheduler_factor != 1.0:
            scheduler_metric = (
                    val_stats["loss_total"]
            )

            scheduler.step(scheduler_metric)

        if logger is not None:
            logger.info(
                "epoch=%03d train_loss=%.6f val_loss=%.6f "
                "train_cfg_mae=%.6f val_cfg_mae=%.6f "
                "train_ee_pos_mae=%.6f val_ee_pos_mae=%.6f "
                "train_ee_rot_mae=%.6f val_ee_rot_mae=%.6f "
                "train_magnet_acc=%.4f val_magnet_acc=%.4f "
                "train_goal_pos_mae=%.4f val_goal_pos_mae=%.4f "
                "train_goal_rot_mae=%.4f val_goal_rot_mae=%.4f "
                "train_obstacle_pos_mae=%.4f val_obstacle_pos_mae=%.4f "
                "train_obstacle_rot_mae=%.4f val_obstacle_rot_mae=%.4f",
                epoch,
                train_stats["loss_total"],
                val_stats["loss_total"],
                train_stats["configuration_mae"],
                val_stats["configuration_mae"],
                train_stats["end_effector_position_mae"],
                val_stats["end_effector_position_mae"],
                train_stats["end_effector_rotation_mae"],
                val_stats["end_effector_rotation_mae"],
                train_stats["magnet_acc"],
                val_stats["magnet_acc"],
                train_stats["goal_obj6D_position_mae"],
                val_stats["goal_obj6D_position_mae"],
                train_stats["goal_obj6D_rotation_mae"],
                val_stats["goal_obj6D_rotation_mae"],
                train_stats["obstacle6D_position_mae"],
                val_stats["obstacle6D_position_mae"],
                train_stats["obstacle6D_rotation_mae"],
                val_stats["obstacle6D_rotation_mae"],
            )

        history["train_loss_total"].append(train_stats["loss_total"])
        history["val_loss_total"].append(val_stats["loss_total"])

        history["train_configuration_mae"].append(train_stats["configuration_mae"])
        history["val_configuration_mae"].append(val_stats["configuration_mae"])

        history["train_end_effector_position_mae"].append(train_stats["end_effector_position_mae"])
        history["val_end_effector_position_mae"].append(val_stats["end_effector_position_mae"])

        history["train_end_effector_rotation_mae"].append(train_stats["end_effector_rotation_mae"])
        history["val_end_effector_rotation_mae"].append(val_stats["end_effector_rotation_mae"])

        history["train_magnet_acc"].append(train_stats["magnet_acc"])
        history["val_magnet_acc"].append(val_stats["magnet_acc"])

        history["train_goal_obj6D_position_mae"].append(train_stats["goal_obj6D_position_mae"])
        history["val_goal_obj6D_position_mae"].append(val_stats["goal_obj6D_position_mae"])

        history["train_goal_obj6D_rotation_mae"].append(train_stats["goal_obj6D_rotation_mae"])
        history["val_goal_obj6D_rotation_mae"].append(val_stats["goal_obj6D_rotation_mae"])

        history["train_obstacle6D_position_mae"].append(train_stats["obstacle6D_position_mae"])
        history["val_obstacle6D_position_mae"].append(val_stats["obstacle6D_position_mae"])

        history["train_obstacle6D_rotation_mae"].append(train_stats["obstacle6D_rotation_mae"])
        history["val_obstacle6D_rotation_mae"].append(val_stats["obstacle6D_rotation_mae"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss_total']:.6f} | "
            f"val_loss={val_stats['loss_total']:.6f} | "
            f"val_configuration_mae={val_stats['configuration_mae']:.6f} | "
            f"val_end_effector_position_mae={val_stats['end_effector_position_mae']:.6f} | "
            f"val_end_effector_rotation_mae={val_stats['end_effector_rotation_mae']:.6f} | "
            f"val_magnet_acc={val_stats['magnet_acc']:.4f} | "
            f"val_goal_obj6D_position_mae={val_stats['goal_obj6D_position_mae']:.6f} | "
            f"val_goal_obj6D_rotation_mae={val_stats['goal_obj6D_rotation_mae']:.6f} | "
            f"val_obstacle6D_position_mae={val_stats['obstacle6D_position_mae']:.6f} | "
            f"val_obstacle6D_rotation_mae={val_stats['obstacle6D_rotation_mae']:.6f}"
        )

        scheduler_dict = None
        if scheduler is not None:
            scheduler_dict = scheduler.state_dict()

        # Checkpoint every epoch, every 10 epoch and for best loss
        if val_stats["loss_total"] < best_val_loss:
            best_val_loss = val_stats["loss_total"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler_dict,
                    "best_val_loss": best_val_loss,
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "history": history,
                },
                Path(checkpoint_dir) / "best_forward_model.pt",
            )

        checkpoint_idx = epoch // 10
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler_dict,
                "best_val_loss": best_val_loss,
                "config": cfg.__dict__,
                "epoch": epoch,
                "history": history,
            },
            Path(checkpoint_dir) / f"forward_model_epoch{checkpoint_idx * 10:03d}.pt",
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler_dict,
                "best_val_loss": best_val_loss,
                "config": cfg.__dict__,
                "epoch": epoch,
                "history": history,
            },
            Path(checkpoint_dir) / f"run_{run_id:03d}.pt",
        )

        i = epoch - 1
        history_row = {
            "run_id": run_id,
            "epoch_id": epoch,
            "val_loss_total": history["val_loss_total"][i],
            "val_configuration_mae": history["val_configuration_mae"][i],
            "val_end_effector_position_mae": history["val_end_effector_position_mae"][i],
            "val_end_effector_rotation_mae": history["val_end_effector_rotation_mae"][i],
            "val_magnet_acc": history["val_magnet_acc"][i],
            "val_goal_obj6D_position_mae": history["val_goal_obj6D_position_mae"][i],
            "val_goal_obj6D_rotation_mae": history["val_goal_obj6D_rotation_mae"][i],
            "val_obstacle6D_position_mae": history["val_obstacle6D_position_mae"][i],
            "val_obstacle6D_rotation_mae": history["val_obstacle6D_rotation_mae"][i],
        }

        plot_metrics(cfg, history, run_id, plots_dir)

        file_exists_and_not_empty = os.path.exists(results_path) and os.path.getsize(results_path) > 0
        if (run_id, epoch) in existing_keys:
            continue

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_row.keys())
            if not file_exists_and_not_empty:
                writer.writeheader()
            writer.writerow(history_row)
        existing_keys.add((run_id, epoch))

    return history

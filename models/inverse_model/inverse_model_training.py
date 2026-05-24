import csv
import os
from logging import Logger
from pathlib import Path
from typing import Dict, List

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from models.helpers import move_batch_to_device, create_optimizer, load_existing_keys
from models.inverse_model.config import InverseModelTrainConfig
from models.inverse_model.loss_function import inverse_model_loss


@torch.no_grad()
def compute_inverse_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Computes evaluation metrics for joint-angle configuration and magnet separately.

    :param prediction: Predicted action vector.
    :param target: Target action vector.
    :return: Dict of metrics.
    """


    metrics: Dict[str, float] = {}

    delta_q_mae = torch.mean(torch.abs(prediction[:, :-1] - target[:, :-1]))
    delta_magnet_mae = torch.mean(torch.abs(prediction[:, -1] - target[:, -1]))

    delta_magnet_prediction = torch.round(prediction[:, -1]).clamp(-1, 1)
    delta_magnet_acc = (delta_magnet_prediction == target[:, -1]).float().mean()

    metrics["delta_q_mae"] = delta_q_mae.item()
    metrics["delta_magnet_mae"] = delta_magnet_mae.item()
    metrics["delta_magnet_acc"] = delta_magnet_acc.item()

    return metrics

def inverse_train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
) -> Dict[str, float]:
    """
    Trains one epoch and keeps track of loss values and evaluation metrics. Model's parameters change.

    :param model: Trained model.
    :param dataloader: Dataloader.
    :param optimizer: Optimiser.
    :param device: Device on which the epoch will run.
    :return: Epoch train metrics.
    """

    model.train()

    running = {
        "loss_total": 0.0,
        "loss_delta_q": 0.0,
        "loss_delta_magnet": 0.0,
        "delta_q_mae": 0.0,
        "delta_magnet_mae": 0.0,
        "delta_magnet_acc": 0.0
    }

    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        target = batch["action"]

        optimizer.zero_grad()

        prediction = model(x)
        loss, loss_stats = inverse_model_loss(prediction, target)

        loss.backward()

        optimizer.step()

        metric_stats = compute_inverse_metrics(prediction, target)

        for key, value in loss_stats.items():
            running[key] += value
        for key, value in metric_stats.items():
            running[key] += value

        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in running.items()}

@torch.no_grad()
def inverse_validate_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
) -> Dict[str, float]:
    """
    Validates the model for one epoch and keeps track of loss values and evaluation metrics.

    :param model: Evaluated model.
    :param dataloader: Dataloader.
    :param device: Device on which the epoch will run.
    :return: Epoch validation metrics.
    """

    model.eval()

    running = {
        "loss_total": 0.0,
        "loss_delta_q": 0.0,
        "loss_delta_magnet": 0.0,
        "delta_q_mae": 0.0,
        "delta_magnet_mae": 0.0,
        "delta_magnet_acc": 0.0
    }

    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        target = batch["action"]

        prediction = model(x)
        loss, loss_stats = inverse_model_loss(prediction, target)
        metric_stats = compute_inverse_metrics(prediction, target)

        for key, value in loss_stats.items():
            running[key] += value
        for key, value in metric_stats.items():
            running[key] += value

        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in running.items()}

def plot_metrics(
        cfg: InverseModelTrainConfig,
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
    plt.title(f"run={run_id} lr={cfg.learning_rate} hidden={cfg.hidden_dimension} n_layer={cfg.n_hidden_layers} patience={cfg.scheduler_patience} factor={cfg.scheduler_factor}")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x_axis, history["val_delta_q_mae"][2:], marker="", label="delta_q_mae")
    plt.plot(x_axis, history["val_delta_magnet_mae"][2:], marker="", label="delta_magnet_mae")
    plt.xlabel("epoch")
    plt.ylabel("val metrics")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x_axis, history["train_delta_q_mae"][2:], marker="", label="delta_q_mae")
    plt.plot(x_axis, history["train_delta_magnet_mae"][2:], marker="", label="delta_magnet_mae")
    plt.xlabel("epoch")
    plt.ylabel("train metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(str(Path(plots_dir) / f"run_{run_id:03d}.png"), bbox_inches="tight")
    plt.close(fig)

def fit_inverse_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: InverseModelTrainConfig,
        logger: Logger,
        plots_dir: str,
        results_path: str,
        checkpoint_dir: str = "best_inverse_model.pt",
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
        "train_delta_q_mae": [],
        "val_delta_q_mae": [],
        "train_delta_magnet_mae": [],
        "val_delta_magnet_mae": [],
        "train_delta_magnet_acc": [],
        "val_delta_magnet_acc": []
    }

    best_val_loss = float("inf")
    start_epoch = 1

    # Load from checkpoint if exists
    checkpoint_path = Path(checkpoint_dir) / f"run_{run_id:03d}.pt"
    if Path(checkpoint_path).exists():
        optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
        history = torch.load(checkpoint_path)["history"]
        best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
        start_epoch = torch.load(checkpoint_path)["epoch"] + 1
        scheduler_dict = torch.load(checkpoint_path).get("scheduler_state_dict", None)
        if scheduler_dict is not None and scheduler is not None:
            scheduler.load_state_dict(scheduler_dict)

    plot_metrics(cfg, history, run_id, plots_dir)
    existing_keys = load_existing_keys(results_path)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_stats = inverse_train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=cfg.device
        )

        val_stats = inverse_validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=cfg.device,
        )

        if scheduler is not None and cfg.scheduler_factor != 1.0:
            scheduler_metric = (
                    val_stats["loss_total"]
            )

            scheduler.step(scheduler_metric)

        if logger is not None:
            logger.info(
                "epoch=%03d train_loss=%.6f val_loss=%.6f "
                "train_delta_q_mae=%.6f val_delta_q_mae=%.6f "
                "train_delta_magnet_mae=%.6f val_delta_magnet_mae=%.6f "
                "train_delta_magnet_acc=%.6f val_delta_magnet_acc=%.6f ",
                epoch,
                train_stats["loss_total"],
                val_stats["loss_total"],
                train_stats["delta_q_mae"],
                val_stats["delta_q_mae"],
                train_stats["delta_magnet_mae"],
                val_stats["delta_magnet_mae"],
                train_stats["delta_magnet_acc"],
                val_stats["delta_magnet_acc"]
            )

        history["train_loss_total"].append(train_stats["loss_total"])
        history["val_loss_total"].append(val_stats["loss_total"])
        history["train_delta_q_mae"].append(train_stats["delta_q_mae"])
        history["val_delta_q_mae"].append(val_stats["delta_q_mae"])
        history["train_delta_magnet_mae"].append(train_stats["delta_magnet_mae"])
        history["val_delta_magnet_mae"].append(val_stats["delta_magnet_mae"])
        history["train_delta_magnet_acc"].append(train_stats["delta_magnet_acc"])
        history["val_delta_magnet_acc"].append(val_stats["delta_magnet_acc"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss_total']:.6f} | "
            f"val_loss={val_stats['loss_total']:.6f} | "
            f"val_delta_q_mae={val_stats['delta_q_mae']:.6f} | "
            f"val_delta_magnet_mae={val_stats['delta_magnet_mae']:.6f} | "
            f"val_delta_magnet_acc={val_stats['delta_magnet_acc']:.6f}"
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
                Path(checkpoint_dir) / "best_inverse_model.pt",
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
            Path(checkpoint_dir) / f"inverse_model_epoch{checkpoint_idx * 10:03d}.pt",
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
            "val_delta_q_mae": history["val_delta_q_mae"][i],
            "val_delta_magnet_mae": history["val_delta_magnet_mae"][i],
            "val_delta_magnet_acc": history["val_delta_magnet_acc"][i],
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

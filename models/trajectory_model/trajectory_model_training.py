import csv
import os
from logging import Logger
from pathlib import Path
from typing import Dict, Any, List

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from models.forward_model.forward_model import ForwardModel
from models.forward_model.loss_function import quaternion_geodesic_loss
from models.helpers import move_batch_to_device, create_optimizer, load_existing_keys
from models.inverse_model.inverse_model import InverseModel
from models.trajectory_model.config import TrajectoryModelTrainConfig
from models.trajectory_model.loss_function import trajectory_model_loss

@torch.no_grad()
def compute_rectification_mae(
        predicted_trajectory: Dict[str, torch.Tensor],
        rectified_trajectory: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes evaluation metrics for rectification process.

    :param predicted_trajectory: Predicted trajectory of state vectors.
    :param rectified_trajectory: Rectified trajectory of state vectors.
    :return: Dict of metrics.
    """

    metrics = {}

    for key in predicted_trajectory.keys():
        pred = predicted_trajectory[key]
        rect = rectified_trajectory[key]

        l2 = torch.norm(pred - rect, dim=-1).mean()
        if "rotation" in key:
            mae = quaternion_geodesic_loss(pred, rect)
        else:
            mae = torch.mean(torch.abs(pred - rect))
        metrics[f"{key}_rectification_mae"] = mae.item()
        metrics[f"{key}_rectification_l2"] = l2.item()

    return metrics

def compute_tail_metrics(
        predicted_end_effector_position: torch.Tensor
) -> Dict[str, float]:
    """
    Computes tail evaluation metrics - effective length, fractional representation of tail and step size in tail.

    :param predicted_end_effector_position: End-effector positions from predicted trajectory.
    :return: Dict of metrics.
    """
    B, T, _ = predicted_end_effector_position.shape

    deltas = torch.norm(
        predicted_end_effector_position[:, 1:, :] - predicted_end_effector_position[:, :-1, :],
        dim=-1
    )

    tail_start = int((T - 1) * 2 / 3)
    tail_deltas = deltas[:, tail_start:]

    tail_mean_step_size = tail_deltas.mean()

    eps = 1e-3
    tail_fraction_static = (tail_deltas < eps).float().mean()

    consecutive_required = 5
    effective_lengths = []

    for b in range(B):
        d = deltas[b]
        found = False
        for t in range(len(d) - consecutive_required):
            if torch.all(d[t:t + consecutive_required] < eps):
                effective_lengths.append(t)
                found = True
                break
        if not found:
            effective_lengths.append(len(d))

    estimated_effective_length = torch.tensor(effective_lengths, dtype=torch.float32, device=predicted_end_effector_position.device).mean()

    return {
        "tail_mean_step_size": tail_mean_step_size.item(),
        "tail_fraction_static": tail_fraction_static.item(),
        "estimated_effective_length": estimated_effective_length.item(),
    }

def compute_angle_metrics(
        predicted_end_effector_position: torch.Tensor,
        target_initial_end_effector_position: torch.Tensor,
        target_final_end_effector_position: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes angle metrics for the trajectory.

    :param predicted_end_effector_position: End-effector positions from predicted trajectory.
    :param target_initial_end_effector_position: End-effector position from ground-truth initial state.
    :param target_final_end_effector_position: End-effector position from ground-truth goal state.
    :return: Dict of metrics
    """

    pts = torch.cat(
        [
            target_initial_end_effector_position.unsqueeze(1),
            predicted_end_effector_position,
            target_final_end_effector_position.unsqueeze(1),
        ],
        dim=1,
    )

    prev_pts = pts[:, :-2, :]
    mid_pts  = pts[:, 1:-1, :]
    next_pts = pts[:, 2:, :]

    v1 = prev_pts - mid_pts
    v2 = next_pts - mid_pts

    v1_norm = torch.norm(v1, dim=-1)
    v2_norm = torch.norm(v2, dim=-1)

    eps = 1e-8
    denom = (v1_norm * v2_norm).clamp_min(eps)

    cos_theta = (v1 * v2).sum(dim=-1) / denom
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angles_rad = torch.acos(cos_theta)
    angles_deg = angles_rad * (180.0 / torch.pi)

    average_angle = angles_deg.mean()
    minimum_angle = angles_deg.min(dim=1).values.mean()

    return {
        "average_angle_between_points": average_angle.item(),
        "minimum_angle_between_points": minimum_angle.item(),
    }

@torch.no_grad()
def compute_trajectory_metrics(
        predicted_trajectory: Dict[str, torch.Tensor],
        rectified_trajectory: Dict[str, torch.Tensor],
        target_initial_state: Dict[str, torch.Tensor],
        target_final_state: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Computes every trajectory metric mention together with distances to boundary positions and step sizes.

    :param predicted_trajectory: Predicted trajectory of states.
    :param rectified_trajectory: Rectified trajectory of states.
    :param target_initial_state: Ground-truth initial state.
    :param target_final_state: Ground-truth final state.
    :return: Dict of metrics.
    """

    metrics: Dict[str, float] = {}

    metrics.update(compute_rectification_mae(predicted_trajectory, rectified_trajectory))
    metrics.update(compute_tail_metrics(predicted_trajectory["end_effector_position"]))
    predicted_end_effector_position = predicted_trajectory["end_effector_position"]
    target_initial_end_effector_position = target_initial_state["end_effector"][:, :3]
    target_final_end_effector_position = target_final_state["end_effector"][:, :3]
    metrics.update(compute_angle_metrics(
        predicted_end_effector_position,
        target_initial_end_effector_position,
        target_final_end_effector_position,
    ))

    distance_to_initial_position = torch.norm(target_initial_end_effector_position - predicted_end_effector_position[:, 0, :], dim=-1).mean()
    distance_to_goal_position = torch.norm(target_final_end_effector_position - predicted_end_effector_position[:, -1, :], dim=-1).mean()

    points = torch.cat(
        [
            target_initial_end_effector_position.unsqueeze(1),
            predicted_end_effector_position,
            target_final_end_effector_position.unsqueeze(1),
        ],
        dim=1,
    )

    segment_lengths = torch.norm(points[:, 1:, :] - points[:, :-1, :], dim=-1)
    average_spacing_between_points = segment_lengths.mean(dim=-1)
    max_spacing_deviation = torch.abs(
        segment_lengths - average_spacing_between_points.unsqueeze(-1)
    ).max(dim=-1).values

    metrics["distance_to_initial_position"] = distance_to_initial_position.item()
    metrics["distance_to_goal_position"] = distance_to_goal_position.item()
    metrics["average_spacing_between_points"] = average_spacing_between_points.mean().item()
    metrics["max_spacing_deviation"] = max_spacing_deviation.mean().item()

    return metrics

@torch.no_grad()
def rectify_trajectory(
        forward_model: ForwardModel,
        inverse_model: InverseModel,
        predicted_trajectory: Dict[str, torch.Tensor],
        initial_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Rectification algorithm.
    """

    batch_size, n_timesteps, _ = predicted_trajectory["configuration"].shape
    device = predicted_trajectory["configuration"].device

    rectified_trajectory = {
        "configuration": torch.zeros(batch_size, n_timesteps, initial_state["configuration"].shape[-1], device=device),
        "end_effector_position": torch.zeros(batch_size, n_timesteps, initial_state["end_effector"][:, :3].shape[-1], device=device),
        "end_effector_rotation": torch.zeros(batch_size, n_timesteps, initial_state["end_effector"][:, 3:].shape[-1],
                                             device=device),
        "magnet": torch.zeros(batch_size, n_timesteps, initial_state["magnet"].shape[-1], device=device),
        "goal_obj6D_position": torch.zeros(batch_size, n_timesteps, initial_state["goal_obj6D"][:, :3].shape[-1], device=device),
        "goal_obj6D_rotation": torch.zeros(batch_size, n_timesteps, initial_state["goal_obj6D"][:, 3:].shape[-1],
                                           device=device),
        "obstacle6D_position": torch.zeros(batch_size, n_timesteps, initial_state["obstacle6D"][:, :3].shape[-1], device=device),
        "obstacle6D_rotation": torch.zeros(batch_size, n_timesteps, initial_state["obstacle6D"][:, 3:].shape[-1], device=device)
    }

    rectified_state = {
        "configuration": initial_state["configuration"],
        "end_effector_position": initial_state["end_effector"][:, :3],
        "end_effector_rotation": initial_state["end_effector"][:, 3:],
        "magnet": initial_state["magnet"],
        "goal_obj6D_position": initial_state["goal_obj6D"][:, :3],
        "goal_obj6D_rotation": initial_state["goal_obj6D"][:, 3:],
        "obstacle6D_position": initial_state["obstacle6D"][:, :3],
        "obstacle6D_rotation": initial_state["obstacle6D"][:, 3:],
    }

    for t in range(n_timesteps):

        predicted_state_t1_without_configuration = torch.cat(
            [
                predicted_trajectory["end_effector_position"][:, t, :],
                predicted_trajectory["end_effector_rotation"][:, t, :],
                predicted_trajectory["goal_obj6D_position"][:, t, :],
                predicted_trajectory["goal_obj6D_rotation"][:, t, :],
                predicted_trajectory["obstacle6D_position"][:, t, :],
                predicted_trajectory["obstacle6D_rotation"][:, t, :]
            ],
            dim=-1
        )

        rectified_state_t = torch.cat(
            [
                rectified_state["configuration"],
                rectified_state["end_effector_position"],
                rectified_state["end_effector_rotation"],
                rectified_state["magnet"],
                rectified_state["goal_obj6D_position"],
                rectified_state["goal_obj6D_rotation"],
                rectified_state["obstacle6D_position"],
                rectified_state["obstacle6D_rotation"],
            ],
            dim=-1
        )

        inverse_model_input = torch.cat(
            [
                rectified_state_t,
                predicted_state_t1_without_configuration
            ],
            dim=-1
        )

        predicted_action_t = inverse_model(inverse_model_input)

        forward_model_input = torch.cat(
            [
                rectified_state_t,
                predicted_action_t
            ],
            dim=-1
        )
        predicted_state_t1 = forward_model(forward_model_input)

        for key in rectified_state.keys():
            rectified_state[key] = predicted_state_t1 [key]

            rectified_trajectory[key][:, t, :] = rectified_state[key]

    return rectified_trajectory


def trajectory_train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        forward_model: ForwardModel,
        inverse_model: InverseModel,
        cfg: TrajectoryModelTrainConfig,
        rect_weight:float,
) -> Dict[str, float]:
    """
    Trains one epoch and keeps track of loss values and evaluation metrics. Model's parameters change.

    :param rect_weight: Weight of the rectification loss.
    :param forward_model: Forward model for rectification.
    :param inverse_model: Inverse model for rectification.
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
        "loss_trajectory": 0.0,
        "loss_initial_position": 0.0,
        "loss_goal_position": 0.0,
        "distance_to_initial_position": 0.0,
        "distance_to_goal_position": 0.0,
        "average_spacing_between_points": 0.0,
        "max_spacing_deviation": 0.0,
        "average_angle_between_points": 0.0,
        "minimum_angle_between_points": 0.0,

        "configuration_rectification_mae": 0.0,
        "end_effector_position_rectification_mae": 0.0,
        "end_effector_rotation_rectification_mae": 0.0,
        "magnet_rectification_mae": 0.0,
        "goal_obj6D_position_rectification_mae": 0.0,
        "goal_obj6D_rotation_rectification_mae": 0.0,
        "obstacle6D_position_rectification_mae": 0.0,
        "obstacle6D_rotation_rectification_mae": 0.0,

        "configuration_rectification_l2": 0.0,
        "end_effector_position_rectification_l2": 0.0,
        "end_effector_rotation_rectification_l2": 0.0,
        "magnet_rectification_l2": 0.0,
        "goal_obj6D_position_rectification_l2": 0.0,
        "goal_obj6D_rotation_rectification_l2": 0.0,
        "obstacle6D_position_rectification_l2": 0.0,
        "obstacle6D_rotation_rectification_l2": 0.0,

        "tail_mean_step_size": 0.0,
        "tail_fraction_static": 0.0,
        "estimated_effective_length": 0.0,
    }

    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        initial_state = {
            "configuration": batch["initial_state_configuration"],
            "end_effector": batch["initial_state_end_effector"],
            "magnet": batch["initial_state_magnet"],
            "goal_obj6D": batch["initial_state_goal_obj6D"],
            "obstacle6D": batch["initial_state_obstacle6D"],
        }
        final_state = {
            "configuration": batch["final_state_configuration"],
            "end_effector": batch["final_state_end_effector"],
            "magnet": batch["final_state_magnet"],
            "goal_obj6D": batch["final_state_goal_obj6D"],
            "obstacle6D": batch["final_state_obstacle6D"],
        }

        optimizer.zero_grad()

        predicted_trajectory = model(x)

        # For pretraining and supervised learning
        if rect_weight > 0:
            rectified_trajectory = rectify_trajectory(forward_model, inverse_model, predicted_trajectory, initial_state)
        else:
            rectified_trajectory = {
                "configuration": batch["trajectory_configuration"],
                "end_effector_position": batch["trajectory_end_effector_position"],
                "end_effector_rotation": batch["trajectory_end_effector_rotation"],
                "magnet": batch["trajectory_magnet"],
                "goal_obj6D_position": batch["trajectory_goal_obj6D_position"],
                "goal_obj6D_rotation": batch["trajectory_goal_obj6D_rotation"],
                "obstacle6D_position": batch["trajectory_obstacle6D_position"],
                "obstacle6D_rotation": batch["trajectory_obstacle6D_rotation"],
            }

        loss, loss_stats = trajectory_model_loss(
            cfg.n_timesteps,
            predicted_trajectory,
            rectified_trajectory,
            initial_state,
            final_state,
            rect_weight,
            lambdas=cfg.lambdas,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.gradient_clip,
        )

        optimizer.step()

        metric_stats = compute_trajectory_metrics(predicted_trajectory, rectified_trajectory, initial_state, final_state)

        for key, value in loss_stats.items():
            running[key] += value
        for key, value in metric_stats.items():
            running[key] += value

        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in running.items()}

@torch.no_grad()
def trajectory_validate_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        forward_model: ForwardModel,
        inverse_model: InverseModel,
        cfg: TrajectoryModelTrainConfig,
        rect_weight:float,
) -> Dict[str, float]:
    """
    Validates the model for one epoch and keeps track of loss values and evaluation metrics.

    :param rect_weight: Weight of the rectification loss.
    :param forward_model: Forward model for rectification.
    :param inverse_model: Inverse model for rectification.
    :param model: Evaluated model.
    :param dataloader: Dataloader.
    :param device: Device on which the epoch will run.
    :param cfg: Configuration of the model.
    :return: Epoch validation metrics.
    """
    model.eval()

    running = {
        "loss_total": 0.0,
        "loss_trajectory": 0.0,
        "loss_initial_position": 0.0,
        "loss_goal_position": 0.0,
        "distance_to_initial_position": 0.0,
        "distance_to_goal_position": 0.0,
        "average_spacing_between_points": 0.0,
        "max_spacing_deviation": 0.0,
        "average_angle_between_points": 0.0,
        "minimum_angle_between_points": 0.0,

        "configuration_rectification_mae": 0.0,
        "end_effector_position_rectification_mae": 0.0,
        "end_effector_rotation_rectification_mae": 0.0,
        "magnet_rectification_mae": 0.0,
        "goal_obj6D_position_rectification_mae": 0.0,
        "goal_obj6D_rotation_rectification_mae": 0.0,
        "obstacle6D_position_rectification_mae": 0.0,
        "obstacle6D_rotation_rectification_mae": 0.0,

        "configuration_rectification_l2": 0.0,
        "end_effector_position_rectification_l2": 0.0,
        "end_effector_rotation_rectification_l2": 0.0,
        "magnet_rectification_l2": 0.0,
        "goal_obj6D_position_rectification_l2": 0.0,
        "goal_obj6D_rotation_rectification_l2": 0.0,
        "obstacle6D_position_rectification_l2": 0.0,
        "obstacle6D_rotation_rectification_l2": 0.0,

        "tail_mean_step_size": 0.0,
        "tail_fraction_static": 0.0,
        "estimated_effective_length": 0.0,
    }

    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        initial_state = {
            "configuration": batch["initial_state_configuration"],
            "end_effector": batch["initial_state_end_effector"],
            "magnet": batch["initial_state_magnet"],
            "goal_obj6D": batch["initial_state_goal_obj6D"],
            "obstacle6D": batch["initial_state_obstacle6D"],
        }
        final_state = {
            "configuration": batch["final_state_configuration"],
            "end_effector": batch["final_state_end_effector"],
            "magnet": batch["final_state_magnet"],
            "goal_obj6D": batch["final_state_goal_obj6D"],
            "obstacle6D": batch["final_state_obstacle6D"],
        }

        predicted_trajectory = model(x)

        # For pretraining and supervised learning
        if rect_weight > 0:
            rectified_trajectory = rectify_trajectory(forward_model, inverse_model, predicted_trajectory, initial_state)
        else:
            rectified_trajectory = {
                "configuration": batch["trajectory_configuration"],
                "end_effector_position": batch["trajectory_end_effector_position"],
                "end_effector_rotation": batch["trajectory_end_effector_rotation"],
                "magnet": batch["trajectory_magnet"],
                "goal_obj6D_position": batch["trajectory_goal_obj6D_position"],
                "goal_obj6D_rotation": batch["trajectory_goal_obj6D_rotation"],
                "obstacle6D_position": batch["trajectory_obstacle6D_position"],
                "obstacle6D_rotation": batch["trajectory_obstacle6D_rotation"],
            }

        loss, loss_stats = trajectory_model_loss(
            cfg.n_timesteps,
            predicted_trajectory,
            rectified_trajectory,
            initial_state,
            final_state,
            rect_weight,
            lambdas=cfg.lambdas,
        )

        metric_stats = compute_trajectory_metrics(predicted_trajectory, rectified_trajectory, initial_state, final_state)

        for key, value in loss_stats.items():
            running[key] += value
        for key, value in metric_stats.items():
            running[key] += value

        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in running.items()}

def plot_metrics(
        n_epochs: int,
        cfg: TrajectoryModelTrainConfig,
        history: Dict[str, List[float]],
        run_id: int,
        plots_dir: str,
):
    """
    Visualises evaluation metrics for both training and validation.

    :param n_epochs: Number of epochs.
    :param cfg: Configuration of the model.
    :param history: History of the run.
    :param run_id: Run ID.
    :param plots_dir: Path to the plots/graphs directory.
    """

    fig = plt.figure(figsize=(10, 8))

    n_points = len(history["val_loss_total"])
    x_axis = range(3, n_epochs + 1)

    plt.subplot(3, 1, 1)
    plt.plot(x_axis, history["val_loss_total"][2:], marker="", label="val_loss")
    plt.plot(x_axis, history["train_loss_total"][2:], marker="", label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"run={run_id} gru_d={cfg.d_gru} gru_n={cfg.n_gru} hidden_d={cfg.hidden_head_dimension} n_timesteps={cfg.n_timesteps}")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x_axis, history["val_distance_to_initial_position"][2:], marker="", label="distance_to_initial_position")
    plt.plot(x_axis, history["val_distance_to_goal_position"][2:], marker="", label="distance_to_goal_position")
    plt.plot(x_axis, history["val_average_spacing_between_points"][2:], marker="", label="average_spacing_between_points")
    plt.plot(x_axis, history["val_max_spacing_deviation"][2:], marker="", label="max_spacing_deviation")
    plt.plot(x_axis, history["val_average_angle_between_points"][2:], marker="", label="average_angle_between_points")
    plt.plot(x_axis, history["val_minimum_angle_between_points"][2:], marker="", label="minimum_angle_between_points")
    plt.xlabel("epoch")
    plt.ylabel("train metrics")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x_axis, history["train_distance_to_initial_position"][2:], marker="", label="distance_to_initial_position")
    plt.plot(x_axis, history["train_distance_to_goal_position"][2:], marker="", label="distance_to_goal_position")
    plt.plot(x_axis, history["train_average_spacing_between_points"][2:], marker="", label="average_spacing_between_points")
    plt.plot(x_axis, history["train_max_spacing_deviation"][2:], marker="", label="max_spacing_deviation")
    plt.plot(x_axis, history["train_average_angle_between_points"][2:], marker="", label="average_angle_between_points")
    plt.plot(x_axis, history["train_minimum_angle_between_points"][2:], marker="", label="minimum_angle_between_points")
    plt.xlabel("epoch")
    plt.ylabel("val metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(str(Path(plots_dir) / f"run_{run_id:03d}.png"), bbox_inches="tight")

def get_training_scheduler(
        epoch: int,
) -> Dict[str, float]:
    """
    Returns scheduler that adjusts the rectification loss and learning rate.

    :param epoch: Current epoch.
    :return: Rectification weight and learning rate based on current epoch.
    """
    if epoch < 10:
        return {
            "lr": 1e-4,
            "rect_weight": 1.0
        }
    else:
        return {
            "lr": 1e-4,
            "rect_weight": 1.0
        }

def fit_trajectory_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrajectoryModelTrainConfig,
    forward_model_path: str,
    inverse_model_path: str,
    plots_dir: str,
    logger: Logger,
    results_path: str,
    checkpoint_dir: str = "best_trajectory_model.pt",
    run_id: int = 0,
) -> Dict[str, list]:
    """
    Trains and then validates the model.

    :param inverse_model_path: Path to the inverse model for rectification.
    :param forward_model_path: Path to the forward model for rectification.
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

    # Load internal models
    checkpoint = torch.load(forward_model_path, map_location=cfg.device)
    fm_cfg = checkpoint["config"]
    forward_model = ForwardModel(
        input_dimension=37,
        shared_hidden_dimension=fm_cfg["shared_hidden_dimension"],
        n_shared_hidden_layers=fm_cfg["n_shared_hidden_layers"],
        head_hidden_dimension=fm_cfg["head_hidden_dimension"],
        output_layers=fm_cfg["output_layers"],
        dropout_rate=fm_cfg["dropout_rate"]
    )
    forward_model.load_state_dict(checkpoint["model_state_dict"])
    forward_model.to(cfg.device)
    forward_model.eval()

    for param in forward_model.parameters():
        param.requires_grad = False

    checkpoint = torch.load(inverse_model_path, map_location=cfg.device)
    im_cfg = checkpoint["config"]

    inverse_model = InverseModel(
        input_dimension=50,
        hidden_dimension=im_cfg["hidden_dimension"],
        output_dimension=im_cfg["output_dimension"],
        n_hidden_layer=im_cfg["n_hidden_layers"],
    )
    inverse_model.load_state_dict(checkpoint["model_state_dict"])
    inverse_model.to(cfg.device)
    inverse_model.eval()

    for param in inverse_model.parameters():
        param.requires_grad = False

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
        "train_distance_to_initial_position": [],
        "val_distance_to_initial_position": [],
        "train_distance_to_goal_position": [],
        "val_distance_to_goal_position": [],
        "train_average_spacing_between_points": [],
        "val_average_spacing_between_points": [],
        "train_max_spacing_deviation": [],
        "val_max_spacing_deviation": [],
        "train_average_angle_between_points": [],
        "val_average_angle_between_points": [],
        "train_minimum_angle_between_points": [],
        "val_minimum_angle_between_points": [],

        "train_configuration_rectification_mae": [],
        "train_end_effector_position_rectification_mae": [],
        "train_end_effector_rotation_rectification_mae": [],
        "train_magnet_rectification_mae": [],
        "train_goal_obj6D_position_rectification_mae": [],
        "train_goal_obj6D_rotation_rectification_mae": [],
        "train_obstacle6D_position_rectification_mae": [],
        "train_obstacle6D_rotation_rectification_mae": [],
        "val_configuration_rectification_mae": [],
        "val_end_effector_position_rectification_mae": [],
        "val_end_effector_rotation_rectification_mae": [],
        "val_magnet_rectification_mae": [],
        "val_goal_obj6D_position_rectification_mae": [],
        "val_goal_obj6D_rotation_rectification_mae": [],
        "val_obstacle6D_position_rectification_mae": [],
        "val_obstacle6D_rotation_rectification_mae": [],

        "train_configuration_rectification_l2": [],
        "train_end_effector_position_rectification_l2": [],
        "train_end_effector_rotation_rectification_l2": [],
        "train_magnet_rectification_l2": [],
        "train_goal_obj6D_position_rectification_l2": [],
        "train_goal_obj6D_rotation_rectification_l2": [],
        "train_obstacle6D_position_rectification_l2": [],
        "train_obstacle6D_rotation_rectification_l2": [],
        "val_configuration_rectification_l2": [],
        "val_end_effector_position_rectification_l2": [],
        "val_end_effector_rotation_rectification_l2": [],
        "val_magnet_rectification_l2": [],
        "val_goal_obj6D_position_rectification_l2": [],
        "val_goal_obj6D_rotation_rectification_l2": [],
        "val_obstacle6D_position_rectification_l2": [],
        "val_obstacle6D_rotation_rectification_l2": [],

        "train_tail_mean_step_size": [],
        "train_tail_fraction_static": [],
        "train_estimated_effective_length": [],
        "val_tail_mean_step_size": [],
        "val_tail_fraction_static": [],
        "val_estimated_effective_length": [],
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
        scheduler_dict: Dict[str, Any] = torch.load(checkpoint_path).get("scheduler_state_dict", None)
        if scheduler_dict is not None and scheduler is not None:
            scheduler.load_state_dict(scheduler_dict)

    plot_metrics(start_epoch - 1, cfg, history, run_id, plots_dir)
    existing_keys = load_existing_keys(results_path)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        schedule = get_training_scheduler(epoch)

        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule["lr"]

        train_stats = trajectory_train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=cfg.device,
            forward_model=forward_model,
            inverse_model=inverse_model,
            cfg=cfg,
            rect_weight=schedule["rect_weight"],
        )

        val_stats = trajectory_validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=cfg.device,
            forward_model=forward_model,
            inverse_model=inverse_model,
            cfg=cfg,
            rect_weight=schedule["rect_weight"],
        )

        if scheduler is not None and cfg.scheduler_factor != 1.0:
            scheduler_metric = (
                    val_stats["loss_total"]
            )

            scheduler.step(scheduler_metric)

        if logger is not None:
            logger.info(
                "epoch=%03d train_loss=%.6f val_loss=%.6f "
                "train_distance_to_initial_position=%.6f val_distance_to_initial_position=%.6f "
                "train_distance_to_goal_position=%.6f val_distance_to_goal_position=%.6f "
                "train_average_spacing_between_points=%.6f val_average_spacing_between_points=%.6f "
                "train_max_spacing_deviation=%.4f val_max_spacing_deviation=%.4f "
                "train_average_spacing_between_points=%.6f val_average_spacing_between_points=%.6f "
                "train_tail_mean_step_size=%.4f val_tail_mean_step_size=%.4f",
                epoch,
                train_stats["loss_total"],
                val_stats["loss_total"],
                train_stats["distance_to_initial_position"],
                val_stats["distance_to_initial_position"],
                train_stats["distance_to_goal_position"],
                val_stats["distance_to_goal_position"],
                train_stats["average_spacing_between_points"],
                val_stats["average_spacing_between_points"],
                train_stats["max_spacing_deviation"],
                val_stats["max_spacing_deviation"],
                train_stats["estimated_effective_length"],
                val_stats["estimated_effective_length"],
                train_stats["tail_mean_step_size"],
                val_stats["tail_mean_step_size"]
            )

        history["train_loss_total"].append(train_stats["loss_total"])
        history["val_loss_total"].append(val_stats["loss_total"])
        history["train_distance_to_initial_position"].append(train_stats["distance_to_initial_position"])
        history["val_distance_to_initial_position"].append(val_stats["distance_to_initial_position"])
        history["train_distance_to_goal_position"].append(train_stats["distance_to_goal_position"])
        history["val_distance_to_goal_position"].append(val_stats["distance_to_goal_position"])
        history["train_average_spacing_between_points"].append(train_stats["average_spacing_between_points"])
        history["val_average_spacing_between_points"].append(val_stats["average_spacing_between_points"])
        history["train_max_spacing_deviation"].append(train_stats["max_spacing_deviation"])
        history["val_max_spacing_deviation"].append(val_stats["max_spacing_deviation"])
        history["train_average_angle_between_points"].append(train_stats["average_angle_between_points"])
        history["val_average_angle_between_points"].append(val_stats["average_angle_between_points"])
        history["train_minimum_angle_between_points"].append(train_stats["minimum_angle_between_points"])
        history["val_minimum_angle_between_points"].append(val_stats["minimum_angle_between_points"])

        history["train_configuration_rectification_mae"].append(train_stats["configuration_rectification_mae"])
        history["val_configuration_rectification_mae"].append(val_stats["configuration_rectification_mae"])
        history["train_end_effector_position_rectification_mae"].append(train_stats["end_effector_position_rectification_mae"])
        history["val_end_effector_position_rectification_mae"].append(val_stats["end_effector_position_rectification_mae"])
        history["train_end_effector_rotation_rectification_mae"].append(train_stats["end_effector_rotation_rectification_mae"])
        history["val_end_effector_rotation_rectification_mae"].append(val_stats["end_effector_rotation_rectification_mae"])
        history["train_magnet_rectification_mae"].append(train_stats["magnet_rectification_mae"])
        history["val_magnet_rectification_mae"].append(val_stats["magnet_rectification_mae"])
        history["train_goal_obj6D_position_rectification_mae"].append(train_stats["goal_obj6D_position_rectification_mae"])
        history["val_goal_obj6D_position_rectification_mae"].append(val_stats["goal_obj6D_position_rectification_mae"])
        history["train_goal_obj6D_rotation_rectification_mae"].append(train_stats["goal_obj6D_rotation_rectification_mae"])
        history["val_goal_obj6D_rotation_rectification_mae"].append(val_stats["goal_obj6D_rotation_rectification_mae"])
        history["train_obstacle6D_position_rectification_mae"].append(train_stats["obstacle6D_position_rectification_mae"])
        history["val_obstacle6D_position_rectification_mae"].append(val_stats["obstacle6D_position_rectification_mae"])
        history["train_obstacle6D_rotation_rectification_mae"].append(train_stats["obstacle6D_rotation_rectification_mae"])
        history["val_obstacle6D_rotation_rectification_mae"].append(val_stats["obstacle6D_rotation_rectification_mae"])

        history["train_configuration_rectification_l2"].append(train_stats["configuration_rectification_l2"])
        history["val_configuration_rectification_l2"].append(val_stats["configuration_rectification_l2"])
        history["train_end_effector_position_rectification_l2"].append(train_stats["end_effector_position_rectification_l2"])
        history["val_end_effector_position_rectification_l2"].append(val_stats["end_effector_position_rectification_l2"])
        history["train_end_effector_rotation_rectification_l2"].append(train_stats["end_effector_rotation_rectification_l2"])
        history["val_end_effector_rotation_rectification_l2"].append(val_stats["end_effector_rotation_rectification_l2"])
        history["train_magnet_rectification_l2"].append(train_stats["magnet_rectification_l2"])
        history["val_magnet_rectification_l2"].append(val_stats["magnet_rectification_l2"])
        history["train_goal_obj6D_position_rectification_l2"].append(train_stats["goal_obj6D_position_rectification_l2"])
        history["val_goal_obj6D_position_rectification_l2"].append(val_stats["goal_obj6D_position_rectification_l2"])
        history["train_goal_obj6D_rotation_rectification_l2"].append(train_stats["goal_obj6D_rotation_rectification_l2"])
        history["val_goal_obj6D_rotation_rectification_l2"].append(val_stats["goal_obj6D_rotation_rectification_l2"])
        history["train_obstacle6D_position_rectification_l2"].append(train_stats["obstacle6D_position_rectification_l2"])
        history["val_obstacle6D_position_rectification_l2"].append(val_stats["obstacle6D_position_rectification_l2"])
        history["train_obstacle6D_rotation_rectification_l2"].append(train_stats["obstacle6D_rotation_rectification_l2"])
        history["val_obstacle6D_rotation_rectification_l2"].append(val_stats["obstacle6D_rotation_rectification_l2"])

        history["train_tail_mean_step_size"].append(train_stats["tail_mean_step_size"])
        history["val_tail_mean_step_size"].append(val_stats["tail_mean_step_size"])
        history["train_tail_fraction_static"].append(train_stats["tail_fraction_static"])
        history["val_tail_fraction_static"].append(val_stats["tail_fraction_static"])
        history["train_estimated_effective_length"].append(train_stats["estimated_effective_length"])
        history["val_estimated_effective_length"].append(val_stats["estimated_effective_length"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss_total']:.6f} | "
            f"val_loss={val_stats['loss_total']:.6f} | "
            f"distance_to_initial={val_stats['distance_to_initial_position']:.6f} | "
            f"distance_to_goal={val_stats['distance_to_goal_position']:.6f} | "
            f"average_spacing={val_stats['average_spacing_between_points']:.6f} | "
            f"average_angle={val_stats['average_angle_between_points']:.4f}"
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
                Path(checkpoint_dir) / f"best_trajectory_model.pt",
            )

        if epoch % 10 == 0:
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
                Path(checkpoint_dir) / f"model_{epoch:04d}.pt",
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
        plot_metrics(epoch, cfg, history, run_id, plots_dir)

        i = epoch - 1
        history_row = {
            "run_id": run_id,
            "epoch_id": epoch,
            "val_loss_total": history["val_loss_total"][i],

            "val_distance_to_initial_position": history["val_distance_to_initial_position"][i],
            "val_distance_to_goal_position": history["val_distance_to_goal_position"][i],
            "val_average_spacing_between_points": history["val_average_spacing_between_points"][i],
            "val_max_spacing_deviation": history["val_max_spacing_deviation"][i],
            "val_average_angle_between_points": history["val_average_angle_between_points"][i],
            "val_minimum_angle_between_points": history["val_minimum_angle_between_points"][i],

            "val_configuration_rectification_mae": history["val_configuration_rectification_mae"][i],
            "val_end_effector_position_rectification_mae": history["val_end_effector_position_rectification_mae"][i],
            "val_end_effector_rotation_rectification_mae": history["val_end_effector_rotation_rectification_mae"][i],
            "val_magnet_rectification_mae": history["val_magnet_rectification_mae"][i],
            "val_goal_obj6D_position_rectification_mae": history["val_goal_obj6D_position_rectification_mae"][i],
            "val_goal_obj6D_rotation_rectification_mae": history["val_goal_obj6D_rotation_rectification_mae"][i],
            "val_obstacle6D_position_rectification_mae": history["val_obstacle6D_position_rectification_mae"][i],
            "val_obstacle6D_rotation_rectification_mae": history["val_obstacle6D_rotation_rectification_mae"][i],

            "val_configuration_rectification_l2": history["val_configuration_rectification_l2"][i],
            "val_end_effector_position_rectification_l2": history["val_end_effector_position_rectification_l2"][i],
            "val_end_effector_rotation_rectification_l2": history["val_end_effector_rotation_rectification_l2"][i],
            "val_magnet_rectification_l2": history["val_magnet_rectification_l2"][i],
            "val_goal_obj6D_position_rectification_l2": history["val_goal_obj6D_position_rectification_l2"][i],
            "val_goal_obj6D_rotation_rectification_l2": history["val_goal_obj6D_rotation_rectification_l2"][i],
            "val_obstacle6D_position_rectification_l2": history["val_obstacle6D_position_rectification_l2"][i],
            "val_obstacle6D_rotation_rectification_l2": history["val_obstacle6D_rotation_rectification_l2"][i],

            "val_tail_mean_step_size": history["val_tail_mean_step_size"][i],
            "val_tail_fraction_static": history["val_tail_fraction_static"][i],
            "val_estimated_effective_length": history["val_estimated_effective_length"][i],
        }

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

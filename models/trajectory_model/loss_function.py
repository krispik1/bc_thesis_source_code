from typing import Tuple, Dict

import torch
import torch.nn.functional as F

from models.forward_model.loss_function import quaternion_chordal_loss

def state_loss(
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        state_idx: int,
        is_target_trajectory: bool = False,
) -> torch.Tensor:
    """
    Loss for rectification/end-point distances. If loss is calculated for a state from trajectory, all terms
    are equal. If loss is calculated for end-point distances, only end-effector position and robot configuration is
    used.

    :param prediction: Predicted state.
    :param target: Rectified or ground truth state.
    :param state_idx: State index in ground truth trajectory.
    :param is_target_trajectory: Flag which loss to calculate - state or distance.
    :return: Total loss normalized.
    """
    t = {}
    lambda_configuration = 1
    lambda_end_effector_position = 1
    lambda_end_effector_rotation = 1
    lambda_magnet = 1
    lambda_goal6D_position = 1
    lambda_goal6D_rotation = 1
    lambda_obstacle6D_position = 1
    lambda_obstacle6D_rotation = 1
    if is_target_trajectory:
        t["configuration"] = target["configuration"][:, state_idx, :]
        t["end_effector_position"] = target["end_effector_position"][:, state_idx, :]
        t["end_effector_rotation"] = target["end_effector_rotation"][:, state_idx, :]
        t["magnet"] = torch.sigmoid(target["magnet"][:, state_idx, :])
        t["goal_obj6D_position"] = target["goal_obj6D_position"][:, state_idx, :]
        t["goal_obj6D_rotation"] = target["goal_obj6D_rotation"][:, state_idx, :]
        t["obstacle6D_position"] = target["obstacle6D_position"][:, state_idx, :]
        t["obstacle6D_rotation"] = target["obstacle6D_rotation"][:, state_idx, :]
    else:
        t["configuration"] = target["configuration"]
        t["end_effector_position"] = target["end_effector"][:, :3]
        t["end_effector_rotation"] = target["end_effector"][:, 3:]
        t["magnet"] = target["magnet"]
        t["goal_obj6D_position"] = target["goal_obj6D"][:, :3]
        t["goal_obj6D_rotation"] = target["goal_obj6D"][:, 3:]
        t["obstacle6D_position"] = target["obstacle6D"][:, :3]
        t["obstacle6D_rotation"] = target["obstacle6D"][:, 3:]

        lambda_configuration = 1
        lambda_end_effector_position = 5
        lambda_end_effector_rotation = 0
        lambda_magnet = 0
        lambda_goal6D_position = 0
        lambda_goal6D_rotation = 0
        lambda_obstacle6D_position = 0
        lambda_obstacle6D_rotation = 0

    normalizer = (
            lambda_configuration
            + lambda_end_effector_position
            + lambda_end_effector_rotation
            + lambda_magnet
            + lambda_goal6D_position
            + lambda_goal6D_rotation
            + lambda_obstacle6D_position
            + lambda_obstacle6D_rotation
    )

    loss_configuration = lambda_configuration *  F.mse_loss(prediction["configuration"][:, state_idx, :], t["configuration"])
    loss_end_effector = (
        lambda_end_effector_position *  F.mse_loss(prediction["end_effector_position"][:, state_idx, :], t["end_effector_position"]) +
        lambda_end_effector_rotation * quaternion_chordal_loss(prediction["end_effector_rotation"][:, state_idx, :], t["end_effector_rotation"])
    )
    loss_magnet = lambda_magnet * F.binary_cross_entropy_with_logits(prediction["magnet"][:, state_idx, :], t["magnet"])
    loss_goal6D = (
        lambda_goal6D_position * F.mse_loss(prediction["goal_obj6D_position"][:, state_idx, :], t["goal_obj6D_position"]) +
        lambda_goal6D_rotation * quaternion_chordal_loss(prediction["goal_obj6D_rotation"][:, state_idx, :], t["goal_obj6D_rotation"])
    )
    loss_obstacle6D = (
        lambda_obstacle6D_position * F.mse_loss(prediction["obstacle6D_position"][:, state_idx, :], t["obstacle6D_position"]) +
        lambda_obstacle6D_rotation * quaternion_chordal_loss(prediction["obstacle6D_rotation"][:, state_idx, :], t["obstacle6D_rotation"])
    )

    loss_total = (loss_configuration + loss_end_effector + loss_magnet + loss_obstacle6D + loss_goal6D) / normalizer

    return loss_total

def acceleration_smoothness_loss(
        ef_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Acceleration smoothness loss - geometric prior to prevent oscillations in safe areas. Oscillation rapidly changes
    distances, and thus acceleration. Heavy zig-zag patterns are penalized.

    :param ef_positions: Positions of the end-effectors.
    :return: Acceleration smoothness loss.
    """
    vel = ef_positions[:, 1:, :] - ef_positions[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]

    return (acc ** 2).mean()

def path_length_loss(
    ef_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Path length loss. Unnecessary movement in safe areas will be punished. Model is encouraged to quickly reach goal
    position.
    :param ef_positions: Positions of the end-effectors.
    :return: Path length loss.
    """

    step = torch.norm(ef_positions[:, 1:] - ef_positions[:, :-1], dim=-1)

    return step.sum(dim=1).mean()

def max_step_loss(
        ef_positions: torch.Tensor,
        max_step: float =0.055
) -> torch.Tensor:
    """
    Maximum step size. Large steps trying to reach the goal position in last few time steps will be penalized.

    :param ef_positions: Positions of the end-effectors.
    :param max_step: Max step allowed estimated from the ground truth datasets analysis.
    :return: Maximum step loss.
    """

    step = torch.norm(ef_positions[:, 1:, :] - ef_positions[:, :-1, :], dim=-1)
    excess = torch.relu(step - max_step)

    return (excess ** 2).mean()

def angle_smoothness_loss(
        ef_positions: torch.Tensor,
        eps: float =1e-8
) -> torch.Tensor:
    """
    Angle smoothness loss. Oscillations result in sharp angles which this error discourages.

    :param ef_positions: Positions of the end-effectors.
    :param eps: Prevention for division by 0 if vectors small.
    :return: Angle smoothness loss.
    """

    v1 = ef_positions[:, 1:-1, :] - ef_positions[:, :-2, :]
    v2 = ef_positions[:, 2:, :] - ef_positions[:, 1:-1, :]

    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)

    cos = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)

    return (1.0 - cos).mean()

def trajectory_model_loss(
        n_timesteps: int,
        predicted_trajectory: Dict[str, torch.Tensor],
        rectified_trajectory: Dict[str, torch.Tensor],
        target_initial_state: Dict[str, torch.Tensor],
        target_final_state: Dict[str, torch.Tensor],
        rectification_weight: float,
        lambdas: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Trajectory model loss. The model is through weights encouraged to reach the goal position. Optional geometric
    priors are available. If a mode of training is supervised, rectified trajectory is instead the ground truth trajectory
    from training dataset.

    :param n_timesteps: Horizon of the model.
    :param predicted_trajectory: Predicted trajectory.
    :param rectified_trajectory: Rectified trajectory.
    :param target_initial_state: Ground truth initial state.
    :param target_final_state: Ground truth final state.
    :param rectification_weight: Weight of rectification, which possibly changes during training.
    :param lambdas: Weights of losses.
    :return: Total loss and values of individual losses.
    """

    loss_trajectory : torch.Tensor = torch.tensor(0.0, device=predicted_trajectory["configuration"].device)
    for i in range(n_timesteps):
        loss_trajectory += state_loss(predicted_trajectory, rectified_trajectory, i, True)
    loss_trajectory /= n_timesteps

    loss_initial_state = 0.5 *  state_loss(predicted_trajectory, target_initial_state, 0)
    loss_final_state = 3.0 * state_loss(predicted_trajectory, target_final_state, -1)
    loss_acc = acceleration_smoothness_loss(predicted_trajectory["end_effector_position"])
    loss_len = path_length_loss(predicted_trajectory["end_effector_position"])
    loss_max_step = max_step_loss(predicted_trajectory["end_effector_position"])
    loss_angle = angle_smoothness_loss(predicted_trajectory["end_effector_position"])

    loss_total = (
            rectification_weight * loss_trajectory +
            loss_initial_state +
            loss_final_state +
            lambdas.get("acceleration", 0.0) * loss_acc +
            lambdas.get("step", 0.0) * loss_max_step +
            lambdas.get("length", 0.0) * loss_len +
            lambdas.get("angle", 0.0) * loss_angle
    )

    statistics = {
        "loss_total": loss_total.item(),
        "loss_trajectory": loss_trajectory .item(),
        "loss_initial_position": loss_initial_state.item(),
        "loss_goal_position": loss_final_state.item()
    }

    return loss_total, statistics

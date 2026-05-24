from typing import Tuple, Dict

import torch
import torch.nn.functional as F

def quaternion_geodesic_loss(q_pred, q_true, eps=1e-7):
    q_pred = F.normalize(q_pred, dim=-1)
    q_true = F.normalize(q_true, dim=-1)

    dot = torch.sum(q_pred * q_true, dim=-1).abs()  # q and -q same rotation
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

    angle = 2.0 * torch.acos(dot)  # radians

    return angle.mean()

def quaternion_chordal_loss(
        q_pred: torch.Tensor,
        q_true: torch.Tensor,
) -> torch.Tensor:
    """
    Loss function for rotational subvector output heads. Uses geometric properties of the quaternions.

    :param q_pred: Predicted rotational quaternion.
    :param q_true: Ground truth quaternion.
    :return: Loss of a rotational head.
    """
    q_pred = F.normalize(q_pred, dim=-1)
    q_true = F.normalize(q_true, dim=-1)

    dot = torch.sum(q_pred * q_true, dim=-1).abs()
    loss = 1.0 - dot

    return loss.mean()

def forward_model_loss(
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        lambdas: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss function for forward model architecture. Calculated for each head separately, final loss is a weighted sum
    of the losses.
    :param prediction: Predicted state.
    :param target: Ground truth state.
    :param lambdas: Loss weights.
    :return: Total loss and separate term values.
    """

    loss_configuration = lambdas.get("configuration", 1.0) *  F.mse_loss(prediction["configuration"], target["configuration"])
    loss_end_effector = (
            lambdas.get("end_effector_position", 1.0) * F.mse_loss(prediction["end_effector_position"], target["end_effector"][:, :3])
            + lambdas.get("end_effector_rotation", 1.0) * quaternion_chordal_loss(prediction["end_effector_rotation"], target["end_effector"][:, 3:])
    )
    loss_magnet = lambdas.get("magnet", 1.0) * F.binary_cross_entropy_with_logits(prediction["magnet"], target["magnet"])
    loss_goal6D = (
            lambdas.get("goal_obj6D_position", 1.0) * F.mse_loss(prediction["goal_obj6D_position"], target["goal_obj6D"][:, :3])
            + lambdas.get("goal_obj6D_rotation", 1.0) * quaternion_chordal_loss(prediction["goal_obj6D_rotation"], target["goal_obj6D"][:, 3:]))
    loss_obstacle6D = (
            lambdas.get("obstacle6D_position", 1.0) * F.mse_loss(prediction["obstacle6D_position"], target["obstacle6D"][:, :3])
            + lambdas.get("obstacle6D_rotation", 1.0) * quaternion_chordal_loss(prediction["obstacle6D_rotation"], target["obstacle6D"][:, 3:]))

    loss_total = loss_configuration + loss_end_effector + loss_magnet + loss_obstacle6D + loss_goal6D
    loss_total /= 8


    statistics = {
        "loss_total": loss_total.item(),
        "loss_configuration": loss_configuration.item(),
        "loss_end_effector": loss_end_effector.item(),
        "loss_magnet": loss_magnet.item(),
        "loss_goal_obj6D": loss_goal6D.item(),
        "loss_obstacle6D": loss_obstacle6D.item(),
    }

    return loss_total, statistics
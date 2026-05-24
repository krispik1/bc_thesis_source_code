from typing import Tuple, Dict

import torch
import torch.nn.functional as F


def inverse_model_loss(
        prediction: torch.Tensor,
        target: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss function for the inverse model architecture. Configuration component of the action vector is heavily weighted.

    :param prediction: Predicted action.
    :param target: Ground truth action.
    :return: Loss values and values of individual terms.
    """
    loss_delta_q = F.mse_loss(prediction[:, :-1], target[:, :-1])
    loss_delta_mgt = F.mse_loss(prediction[:, -1], target[:, -1])

    loss_total = 7 * loss_delta_q + loss_delta_mgt

    statistics = {
        "loss_total": loss_total.item(),
        "loss_delta_q": loss_delta_q.item(),
        "loss_delta_magnet": loss_delta_mgt.item(),
    }

    return loss_total, statistics
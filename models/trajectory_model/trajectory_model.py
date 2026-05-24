from typing import Dict, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class TrajectoryModel(nn.Module):
    def __init__(
            self,
            input_dimension: int,
            n_gru: int,
            dimension_gru: int,
            output_layers: Dict[str, int],
            device: str,
            hidden_head_dimension: int,
            n_timesteps: int
    ) -> None:

        super().__init__()

        self.n_timesteps = n_timesteps
        self.input_dimension = input_dimension
        self.n_gru = n_gru
        self.d_gru = dimension_gru
        self.head_hidden_dimension = hidden_head_dimension
        self.output_layers = output_layers

        self.device = device

        self.grus = nn.ModuleList(
            [
                nn.GRUCell(
                    input_size=self.input_dimension,
                    hidden_size=self.d_gru,
                    device=self.device
                )
            ] + [
                nn.GRUCell(
                    input_size=self.d_gru,
                    hidden_size=self.d_gru,
                    device=self.device
                )
                for _ in range(self.n_gru - 1)
            ]
        )

        self.shared = nn.Linear(self.d_gru, self.d_gru, device=self.device)

        self.output_head_layers = nn.ModuleDict({
            output_name: (
                nn.Sequential(
                    nn.Linear(self.d_gru, hidden_head_dimension, device=self.device),
                    nn.Tanh(),
                    nn.Linear(hidden_head_dimension, output_dimension, device=self.device)
                )
            )
            for output_name, output_dimension in output_layers.items()
        })

    def forward(
            self,
            x: Tensor,
    ) -> Dict[str, Tensor]:

        batch_size, _ = x.shape

        h_grus: List[Tensor] = [
            torch.zeros(batch_size, self.d_gru, device=self.device)
            for _ in range(self.n_gru)
        ]
        outputs: Dict[str, Tensor] = {
            output_name: torch.zeros(batch_size, self.n_timesteps, output_dimension, device=self.device)
            for output_name, output_dimension in self.output_layers.items()
        }

        for t in range(self.n_timesteps):
            h = x
            for gru_id, gru in enumerate(self.grus):
                h = gru(h, h_grus[gru_id])
                h_grus[gru_id] = h

            h = self.shared(F.tanh(h))

            for output_name, output_head in self.output_head_layers.items():
                outputs[output_name][:, t, :] = output_head(h)

        return outputs
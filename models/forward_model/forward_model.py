from typing import Dict

from torch import nn, Tensor

class ForwardModel(nn.Module):

    def __init__(
            self,
            input_dimension: int,
            shared_hidden_dimension: int,
            n_shared_hidden_layers: int,
            head_hidden_dimension: int,
            output_layers: Dict[str, int],
            dropout_rate: float,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.shared_hidden_dimension = shared_hidden_dimension
        self.head_hidden_dimension = head_hidden_dimension
        self.output_head_layers = output_layers

        self.shared_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dimension, shared_hidden_dimension),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate),
                )
            ] + [
                nn.Sequential(
                    nn.Linear(shared_hidden_dimension, shared_hidden_dimension),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(n_shared_hidden_layers - 1)
            ]
        )

        self.output_head_layers = nn.ModuleDict({
            output_name :
                nn.Sequential(
                    nn.Linear(shared_hidden_dimension, head_hidden_dimension),
                    nn.Tanh(),
                    nn.Linear(head_hidden_dimension, output_dimension),
                )
            for output_name, output_dimension in output_layers.items()
        })

    def forward(
            self,
            x: Tensor
    ) -> Dict[str, Tensor]:

        h = x
        for shared_layer in self.shared_layers:
            h = shared_layer(h)

        outputs: Dict[str, Tensor] = {}
        for output_head_name, output_head in self.output_head_layers.items():
            outputs[output_head_name] = output_head(h)

        return outputs
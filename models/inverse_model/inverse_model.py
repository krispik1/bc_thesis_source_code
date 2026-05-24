from torch import nn, Tensor

class InverseModel(nn.Module):

    def __init__(
            self,
            input_dimension: int,
            hidden_dimension: int,
            output_dimension: int,
            n_hidden_layer: int
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layer = n_hidden_layer

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dimension, self.hidden_dimension),
                    nn.Tanh(),
                )
            ] + [
                nn.Sequential(
                    nn.Linear(self.hidden_dimension, self.hidden_dimension),
                    nn.Tanh(),
                )
                for _ in range(self.n_hidden_layer - 1)
            ] + [
                nn.Sequential(
                    nn.Linear(self.hidden_dimension, self.output_dimension)
                )
            ]
        )

    def forward(
            self,
            x: Tensor
    ) -> Tensor:

        for layer in self.hidden_layers:
            x = layer(x)
        return x
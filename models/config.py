from dataclasses import dataclass

@dataclass
class ModelTrainConfig:
    # Superclass of the model configs
    model_name: str

    num_epochs: int
    early_stopping_patience: int
    device: str

    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
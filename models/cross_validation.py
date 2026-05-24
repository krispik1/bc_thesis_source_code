import csv
import os
from logging import Logger
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader



from remake.models.config import ModelTrainConfig
from remake.models.create_datasets import trajectory_k_fold_cross_validation_dataset_indices, \
    internal_k_fold_cross_validation_dataset_indices
from remake.models.datasets import PretrainTrajectoryModelDataset, InverseModelDataset, ForwardModelDataset
from remake.models.forward_model.config import ForwardModelTrainConfig
from remake.models.forward_model.forward_model import ForwardModel
from remake.models.forward_model.forward_model_training import fit_forward_model
from remake.models.helpers import setup_logger, load_existing_keys
from remake.models.inverse_model.config import InverseModelTrainConfig
from remake.models.inverse_model.inverse_model import InverseModel
from remake.models.inverse_model.inverse_model_training import fit_inverse_model
from remake.models.trajectory_model.config import TrajectoryModelTrainConfig
from remake.models.trajectory_model.trajectory_model import TrajectoryModel
from remake.models.trajectory_model.trajectory_model_training import fit_trajectory_model


def cross_validation(
        cfg_idx: int,
        cfg: ModelTrainConfig,
        data_path: str,
        index_file_path: str,
        log_dir: str,
        checkpoint_dir: str,
        results_dir: str,
        plots_dir: str,
        forward_model_path: str,
        inverse_model_path: str,
        n_folds: int = 5,
):
    """
    Method used for k-fold cross-validation of the model configuration.

    :param cfg_idx: Index of the model configuration.
    :param cfg: Configuration of the model.
    :param data_path: Dataset used for cross-validation.
    :param index_file_path: Index file for the data file.
    :param log_dir: Log folder.
    :param checkpoint_dir: Checkpoint folder.
    :param results_dir: Results folder.
    :param plots_dir: Plots folder.
    :param forward_model_path: Forward model path if needed.
    :param inverse_model_path: Inverse model path if needed.
    :param n_folds: Number of folds.
    :return:
    """

    splits = get_k_fold_cross_validation_dataset_indices(
        index_file_path=index_file_path,
        n_folds=n_folds,
        model_name=cfg.model_name,
    )

    logger = setup_logger(log_dir, f"{cfg.model_name}_model")

    for k in range(n_folds):

        train_loader, val_loader = get_data_loader(
            cfg=cfg,
            data_path=data_path,
            run_id=k,
            n_folds=n_folds,
            splits=splits
        )

        checkpoint_path = str(Path(checkpoint_dir) / f"run_{k + 1:03d}.pt")

        model = load_model(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            logger=logger,
            run_id=k + 1
        )

        history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_dir,
            forward_model_path=forward_model_path,
            inverse_model_path=inverse_model_path,
            logger=logger,
            run_id=k+1,
            plots_dir=plots_dir,
            results_dir=str(Path(results_dir) / f"{cfg.model_name}_sweep_histories.csv"),
        )

        row = {
            "run_id": k + 1,
            "epoch_id": cfg.num_epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "optimizer": cfg.optimizer,

            "checkpoint_path": checkpoint_path,
        }
        row.update(get_run_row(history, cfg))

        results_path = Path(results_dir) / f"{cfg.model_name}_progress{cfg_idx}.csv"
        file_exists_and_not_empty = os.path.exists(results_path) and os.path.getsize(results_path) > 0
        keys = load_existing_keys(str(results_path))
        if (k, cfg.num_epochs) in keys:
            continue

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists_and_not_empty:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Finished run=%d last_val_loss=%.6f.", k + 1, row["last_val_loss"])

def load_model(
        cfg: ModelTrainConfig,
        checkpoint_path: str,
        logger: Logger,
        run_id: int,
) -> nn.Module:
    """
    Creates a model based on config, logs start of run. If the model has a checkpoint available, load it.

    :param cfg: Config of the model - forward, inverse or trajectory.
    :param checkpoint_path: Possible checkpoint path.
    :param logger: Logger object.
    :param run_id: Run ID.
    :return: Loaded model.
    """

    if isinstance(cfg, TrajectoryModelTrainConfig):
        model = TrajectoryModel(
            input_dimension=cfg.input_dimension,
            n_gru=cfg.n_gru,
            dimension_gru=cfg.d_gru,
            hidden_head_dimension=cfg.hidden_head_dimension,
            output_layers=cfg.output_layers,
            device=cfg.device,
            n_timesteps=cfg.n_timesteps
        )

        logger.info(
            "Starting run=%03d optimizer=%s lr=%g n_timesteps=%d hidden=%d amount=%d head=%d.",
            run_id + 1, cfg.optimizer, cfg.learning_rate, cfg.n_timesteps, cfg.d_gru, cfg.n_gru, cfg.hidden_head_dimension,
        )
    elif isinstance(cfg, InverseModelTrainConfig):
        model = InverseModel(
            input_dimension=cfg.input_dimension,
            hidden_dimension=cfg.hidden_dimension,
            output_dimension=cfg.output_dimension,
            n_hidden_layer=cfg.n_hidden_layers,
        )

        logger.info(
            "Starting run=%03d lr=%g batch=%d hidden=%d decay=%g.",
            run_id + 1, cfg.learning_rate, cfg.batch_size, cfg.hidden_dimension,
            cfg.weight_decay,
        )
    elif isinstance(cfg, ForwardModelTrainConfig):
        model = ForwardModel(
            input_dimension=cfg.input_dimension,
            shared_hidden_dimension=cfg.shared_hidden_dimension,
            head_hidden_dimension=cfg.head_hidden_dimension,
            output_layers=cfg.output_layers,
            n_shared_hidden_layers=cfg.n_shared_hidden_layers,
            dropout_rate=cfg.dropout_rate
        )

        logger.info(
            "Starting run=%03d lr=%g batch=%d shared=%d head=%d decay=%g.",
            run_id+1, cfg.learning_rate, cfg.batch_size, cfg.shared_hidden_dimension, cfg.head_hidden_dimension,
            cfg.weight_decay,
        )
    else:
        raise ValueError("Unknown model type")

    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    return model

def get_k_fold_cross_validation_dataset_indices(
        index_file_path: str,
        n_folds: int,
        model_name: str
) -> List[List[int]]:
    """
    Returns data indices fold splits based on model type.

    :param index_file_path: File of indices to the data.
    :param n_folds: Number of folds.
    :param model_name: Model name.
    :return: Folds as splits of indices.
    """

    if model_name == "trajectory":
        return trajectory_k_fold_cross_validation_dataset_indices(
            master_index_df_path=index_file_path,
            k=n_folds
        )
    elif model_name == "inverse":
        return internal_k_fold_cross_validation_dataset_indices(
            master_index_df_path=index_file_path,
            k=n_folds
        )
    elif model_name == "forward":
        return internal_k_fold_cross_validation_dataset_indices(
            master_index_df_path=index_file_path,
            k=n_folds
        )
    else:
        raise ValueError("Unknown model type")

def get_data_loader(
        cfg: ModelTrainConfig,
        data_path: str,
        run_id: int,
        n_folds: int,
        splits: List[List[int]],
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns data loaders for current fold based on model.

    :param cfg: Configuration of the models - forward, inverse or trajectory.
    :param data_path: Path of the file containing the dataset.
    :param run_id: Run ID.
    :param n_folds: Number of folds.
    :param splits: Splits of indices pointing to data points.
    :return: Data loaders for current fold based on the model.
    """
    train_indices = []
    for i in range(n_folds):
        if i == run_id:
            continue
        train_indices += splits[i]

    val_indices = splits[run_id]

    if isinstance(cfg, TrajectoryModelTrainConfig):
        train_dataset = PretrainTrajectoryModelDataset(data_path, train_indices)
        val_dataset = PretrainTrajectoryModelDataset(data_path, val_indices)
    elif isinstance(cfg, InverseModelTrainConfig):
        train_dataset = InverseModelDataset(data_path, train_indices)
        val_dataset = InverseModelDataset(data_path, val_indices)
    elif isinstance(cfg, ForwardModelTrainConfig):
        train_dataset = ForwardModelDataset(data_path, train_indices)
        val_dataset = ForwardModelDataset(data_path, val_indices)
    else:
        raise ValueError("Unknown model type")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=10)

    return train_loader, val_loader

def fit_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ModelTrainConfig,
        checkpoint_path: str,
        forward_model_path: str,
        inverse_model_path: str,
        logger: Logger,
        run_id: int,
        plots_dir: str,
        results_dir: str,
) -> Dict[str, Any]:
    """
    Training and evaluation loop.

    :param model: Trained model.
    :param train_loader: Train data loader.
    :param val_loader: Validation data loader.
    :param cfg: Config of the model - forward, inverse or trajectory.
    :param checkpoint_path: Checkpoint path.
    :param forward_model_path: Forward model path if needed.
    :param inverse_model_path: Inverse model path if needed.
    :param logger: Logger object.
    :param run_id: Run ID.
    :param plots_dir: Plot directory.
    :param results_dir: Results directory.
    :return: History of training and validation metrics.
    """

    if isinstance(cfg, TrajectoryModelTrainConfig):
        return fit_trajectory_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            forward_model_path=forward_model_path,
            inverse_model_path=inverse_model_path,
            logger=logger,
            run_id=run_id+1,
            plots_dir=plots_dir,
            results_path=str(Path(results_dir) / f"trajectory_model_sweep_histories.csv"),
        )
    elif isinstance(cfg, InverseModelTrainConfig):
        return fit_inverse_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            plots_dir=plots_dir,
            logger=logger,
            k=run_id+1,
            results_path=str(Path(results_dir) / "inverse_model_history.csv")
        )
    elif isinstance(cfg, ForwardModelTrainConfig):
        return fit_forward_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            plots_dir=plots_dir,
            logger=logger,
            k=run_id+1,
            results_path=str(Path(results_dir) / "forward_model_history.csv")
        )
    else:
        raise ValueError("Unknown model type")

def get_run_row(
        history: Dict[str, Any],
        cfg: ModelTrainConfig,
) -> Dict[str, Any]:
    """
    Based on history and model, returns row of data for that model.

    :param history: History of training and validation metrics.
    :param cfg: Config of the model - forward, inverse or trajectory.
    :return: Dict representing row of data for that model.
    """

    if isinstance(cfg, TrajectoryModelTrainConfig):
        return {
            "n_timesteps": cfg.n_timesteps,
            "n_gru": cfg.n_gru,
            "d_gru": cfg.d_gru,
            "hidden_head_dimension": cfg.hidden_head_dimension,
            "scheduler_factor": cfg.scheduler_factor,
            "scheduler_patience": cfg.scheduler_patience,
            "grad_clip": cfg.gradient_clip,

            "lambda_acc": cfg.lambdas.get("acceleration", 0.0),
            "lambda_max_step": cfg.lambdas.get("step", 0.0),
            "lambda_len": cfg.lambdas.get("length", 0.0),
            "lambda_angle": cfg.lambdas.get("angle", 0.0),

            "last_val_loss": history["val_loss_total"][-1],
            "last_val_distance_to_initial_position": history["val_distance_to_initial_position"][-1],
            "last_val_distance_to_goal_position": history["val_distance_to_goal_position"][-1],
            "last_val_average_spacing_between_points": history["val_average_spacing_between_points"][-1],
            "last_val_average_angle_between_points": history["val_average_angle_between_points"][-1],
        }
    elif isinstance(cfg, InverseModelTrainConfig):
        return {
            "hidden_dimension": cfg.hidden_dimension,
            "n_layers": cfg.n_hidden_layers,

            "last_val_loss": history["val_loss_total"][-1],
            "last_val_delta_q_mae": history["val_delta_q_mae"][-1],
            "last_val_delta_magnet_mae": history["val_delta_magnet_mae"][-1],
            "last_val_delta_magnet_acc": history["val_delta_magnet_acc"][-1],
        }
    elif isinstance(cfg, ForwardModelTrainConfig):
        return {
            "shared_hidden_dimension": cfg.shared_hidden_dimension,
            "head_hidden_dimension": cfg.head_hidden_dimension,
            "n_layers": cfg.n_shared_hidden_layers,

            "last_val_loss_total": history["val_loss_total"][-1],
            "last_val_configuration_mae": history["val_configuration_mae"][-1],
            "last_val_end_effector_position_mae": history["val_end_effector_position_mae"][-1],
            "last_val_end_effector_rotation_mae": history["val_end_effector_rotation_mae"][-1],
            "last_val_magnet_acc": history["val_magnet_acc"][-1],
            "last_val_goal_obj6D_position_mae": history["val_goal_obj6D_position_mae"][-1],
            "last_val_goal_obj6D_rotation_mae": history["val_goal_obj6D_rotation_mae"][-1],
            "last_val_obstacle6D_position_mae": history["val_obstacle6D_position_mae"][-1],
            "last_val_obstacle6D_rotation_mae": history["val_obstacle6D_rotation_mae"][-1],
        }
    else:
        raise ValueError("Unknown model type")
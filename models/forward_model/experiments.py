import csv
import os
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from remake.models.forward_model.config import ForwardModelTrainConfig
from remake.models.forward_model.forward_model import ForwardModel
from remake.models.forward_model.forward_model_training import fit_forward_model
from remake.models.helpers import setup_logger, load_existing_keys


def run_forward_model_experiments(
        train_dataset: Dataset,
        val_dataset: Dataset,
        results_dir: str,
        checkpoint_dir: str,
        log_dir: str,
        plots_dir:str,
        after_run_callback = None,
):
    """
    Runs experiments defined in the body of the function.

    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param results_dir: Path to the results directory.
    :param checkpoint_dir: Path to the checkpoint directory.
    :param log_dir: Path to the logs directory.
    :param plots_dir: Path to the plots/graphs directory.
    :param after_run_callback: Function to be called after each run.
    """

    lrs = [1e-4]
    shareds = [512, 768, 1024]
    heads = [128]
    layers = [1]

    run_id = 0
    logger = setup_logger(log_dir, "forward_model")

    # Create hyperparameter grid
    for lr, head, shared, layer in product(
        lrs, heads, shareds, layers
    ):

        run_id += 1

        # Create config for forward model
        cfg = ForwardModelTrainConfig(
            num_epochs=100,
            shared_hidden_dimension=shared,
            head_hidden_dimension=head,
            learning_rate=lr,
            n_shared_hidden_layers=layer,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=6,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=6,
        )

        # Initialise the model
        model = ForwardModel(
            input_dimension=cfg.input_dimension,
            shared_hidden_dimension=cfg.shared_hidden_dimension,
            n_shared_hidden_layers=cfg.n_shared_hidden_layers,
            head_hidden_dimension=cfg.head_hidden_dimension,
            output_layers=cfg.output_layers,
            dropout_rate=cfg.dropout_rate,
        )

        checkpoint_path = str(Path(checkpoint_dir) / f"run_{run_id:03d}.pt")

        # Check if checkpoint exists and load weights
        if Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

        logger.info(
            "Starting run=%03d lr=%g shared=%d head=%d layers=%d.",
            run_id, cfg.learning_rate, cfg.shared_hidden_dimension, cfg.head_hidden_dimension, cfg.n_shared_hidden_layers,
        )

        # Train the model
        history = fit_forward_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_dir,
            logger=logger,
            k=run_id,
            plots_dir=plots_dir,
            results_path=str(Path(results_dir) / f"forward_model_sweep_histories.csv"),
        )


        best_val_loss = min(history["val_loss_total"])
        best_epoch_idx = history["val_loss_total"].index(best_val_loss)

        # Save statistics from the run
        runs_row = {
                "run_id": run_id,
                "num_epoch": cfg.num_epochs,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "optimizer": cfg.optimizer,
                "shared_hidden_dimension": cfg.shared_hidden_dimension,
                "head_hidden_dimension": cfg.head_hidden_dimension,
                "dropout_rate": cfg.dropout_rate,
                "n_layers": cfg.n_shared_hidden_layers,
                "scheduler_factor": cfg.scheduler_factor,
                "scheduler_patience": cfg.scheduler_patience,

                "lambda_configuration": 1.0,
                "lambda_ee_position": 1.0,
                "lambda_ee_rotation": 1.0,
                "lamba_magnet": 1.0,
                "lambda_goal_position": 1.0,
                "lambda_goal_rotation": 1.0,
                "lambda_occ_position": 1.0,
                "lambda_occ_rotation": 1.0,

                "best_epoch": best_epoch_idx + 1,
                "best_val_loss": best_val_loss,
                "best_val_configuration_mae": history["val_configuration_mae"][best_epoch_idx],
                "best_val_end_effector_position_mae": history["val_end_effector_position_mae"][best_epoch_idx],
                "best_val_end_effector_rotation_mae": history["val_end_effector_rotation_mae"][best_epoch_idx],
                "best_val_magnet_acc": history["val_magnet_acc"][best_epoch_idx],
                "best_val_goal_obj6D_position_mae": history["val_goal_obj6D_position_mae"][best_epoch_idx],
                "best_val_goal_obj6D_rotation_mae": history["val_goal_obj6D_rotation_mae"][best_epoch_idx],
                "best_val_obstacle6D_position_mae": history["val_obstacle6D_position_mae"][best_epoch_idx],
                "best_val_obstacle6D_rotation_mae": history["val_obstacle6D_rotation_mae"][best_epoch_idx],
            }

        logger.info("Finished run=%03d best_val_loss=%.6f.", run_id, min(history["val_loss_total"]))

        results_path = Path(results_dir) / "forward_model_sweep_runs.csv"
        file_exists_and_not_empty = os.path.exists(results_path) and os.path.getsize(results_path) > 0
        keys = load_existing_keys(str(results_path))
        if (run_id, cfg.num_epochs) in keys:
            continue

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=runs_row.keys())
            if not file_exists_and_not_empty:
                writer.writeheader()
            writer.writerow(runs_row)

        if after_run_callback is not None:
            after_run_callback(str(Path(results_dir) / "forward_model_sweep_runs.csv"))
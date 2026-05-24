import csv
import os
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from models.helpers import setup_logger, load_existing_keys
from models.inverse_model.config import InverseModelTrainConfig
from models.inverse_model.inverse_model import InverseModel
from models.inverse_model.inverse_model_training import fit_inverse_model


def run_inverse_model_experiments(
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

    n_layers = [4]
    learning_rates = [1e-4]
    hidden_dimensions = [1024]
    patiences = [8]
    factors = [1.0]

    run_id = 0

    logger = setup_logger(log_dir, "inverse_model")

    # Create hyperparameter grid
    for n_layer, learning_rate, hidden_dimension, patience, factor in product(
        n_layers, learning_rates, hidden_dimensions, patiences, factors
    ):

        run_id += 1

        # Create config for inverse model
        cfg = InverseModelTrainConfig(
            num_epochs=50,
            scheduler_factor=factor,
            scheduler_patience=patience,
            learning_rate=learning_rate,
            hidden_dimension=hidden_dimension,
            n_hidden_layers=n_layer,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=10,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=10,
        )

        # Initialise the model
        model = InverseModel(
            input_dimension=cfg.input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=cfg.output_dimension,
            n_hidden_layer=cfg.n_hidden_layers,
        )

        checkpoint_path = str(Path(checkpoint_dir) / f"run_{run_id:03d}.pt")

        # Check if checkpoint exists and load weights
        if Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

        logger.info(
            "Starting run=%03d lr=%g hidden=%d patience=%d factor=%g.",
            run_id, learning_rate, hidden_dimension, patience, factor,
        )

        # Train the model
        history = fit_inverse_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_dir,
            logger=logger,
            plots_dir=plots_dir,
            results_path=str(Path(results_dir) / f"inverse_model_sweep_histories.csv"),
            k=run_id,
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
            "hidden_dimension": cfg.hidden_dimension,
            "n_layers": cfg.n_hidden_layers,

            "best_epoch": best_epoch_idx + 1,
            "best_val_loss": best_val_loss,
            "best_val_delta_q_mae": history["val_delta_q_mae"][best_epoch_idx],
            "best_val_delta_magnet_mae": history["val_delta_magnet_mae"][best_epoch_idx],
            "best_val_delta_magnet_acc": history["val_delta_magnet_acc"][best_epoch_idx],
        }

        results_path = Path(results_dir) / "inverse_model_sweep_runs.csv"
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
            after_run_callback(str(Path(results_dir) / "inverse_model_sweep_runs.csv"))

        logger.info("Finished run=%03d best_val_loss=%.6f.", run_id, min(history["val_loss_total"]))

import csv
import os
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from models.helpers import setup_logger, load_existing_keys
from models.trajectory_model.config import TrajectoryModelTrainConfig
from models.trajectory_model.trajectory_model import TrajectoryModel
from models.trajectory_model.trajectory_model_training import fit_trajectory_model


def run_trajectory_model_experiments(
        train_dataset: Dataset,
        val_dataset: Dataset,
        results_dir: str,
        checkpoint_dir: str,
        log_dir: str,
        plots_dir: str,
        forward_model_path: str,
        inverse_model_path: str,
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
    :param forward_model_path: Forward model used for rectification.
    :param inverse_model_path: Inverse model used for rectification.
    :param after_run_callback: Function to be called after each run.
    """

    optimizer_and_learning_rates = {
        "AdamW": [1e-4],
    }

    n_timesteps = [50]
    gru_amounts = [2]
    hidden_gru_dimensions = [768, 1024]
    hidden_head_dimensions = [256]

    run_id = 1
    logger = setup_logger(log_dir, "trajectory_model")

    # Create hyperparameter grid
    for optimizer_and_learning_rate in optimizer_and_learning_rates.items():

        for optimizer, learning_rate, n_timestep, amount, hidden, head in product(
                [optimizer_and_learning_rate[0]], optimizer_and_learning_rate[1], n_timesteps, gru_amounts,
                hidden_gru_dimensions, hidden_head_dimensions
        ):

            run_id += 1

            # Create config for trajectory model
            cfg = TrajectoryModelTrainConfig(
                num_epochs=30,
                learning_rate=learning_rate,
                optimizer=optimizer,
                n_timesteps=n_timestep,
                d_gru=hidden,
                n_gru=amount,
                hidden_head_dimension=head,
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
            model = TrajectoryModel(
                input_dimension=cfg.input_dimension,
                n_gru=cfg.n_gru,
                dimension_gru=cfg.d_gru,
                hidden_head_dimension=cfg.hidden_head_dimension,
                output_layers=cfg.output_layers,
                device=cfg.device,
                n_timesteps=cfg.n_timesteps
            )

            checkpoint_path = str(Path(checkpoint_dir) / f"run_{run_id:03d}.pt")

            # Check if checkpoint exists and load weights
            if Path(checkpoint_path).exists():
                model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

            logger.info(
                "Starting run=%03d optimizer=%s lr=%g n_timesteps=%d hidden=%d amount=%d head=%d.",
                run_id, cfg.optimizer, cfg.learning_rate, cfg.n_timesteps, cfg.d_gru, cfg.n_gru, cfg.hidden_head_dimension,
            )

            # Train the model
            history = fit_trajectory_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                checkpoint_dir=checkpoint_dir,
                forward_model_path=forward_model_path,
                inverse_model_path=inverse_model_path,
                logger=logger,
                run_id=run_id,
                plots_dir=plots_dir,
                results_path=str(Path(results_dir) / f"trajectory_model_sweep_histories.csv"),
            )

            best_val_loss = min(history["val_loss_total"])
            best_epoch_idx = history["val_loss_total"].index(best_val_loss)

            # Save statistics from the run
            row = {
                "run_id": run_id,
                "num_epoch": cfg.num_epochs,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "optimizer": cfg.optimizer,
                "n_timesteps": cfg.n_timesteps,
                "n_gru": cfg.n_gru,
                "d_gru": cfg.d_gru,
                "hidden_head_dimension": cfg.hidden_head_dimension,
                "scheduler_factor": cfg.scheduler_factor,
                "scheduler_patience": cfg.scheduler_patience,
                "grad_clip": cfg.gradient_clip,

                "lambda_acc": 0.001,
                "lambda_max_step": 0.001,
                "lambda_len": 0.001,

                "best_epoch": best_epoch_idx + 1,
                "best_val_loss": best_val_loss,
                "best_val_distance_to_initial_position": history["val_distance_to_initial_position"][best_epoch_idx],
                "best_val_distance_to_goal_position": history["val_distance_to_goal_position"][best_epoch_idx],
                "best_val_average_spacing_between_points": history["val_average_spacing_between_points"][best_epoch_idx],
                "best_val_max_spacing_deviation": history["val_max_spacing_deviation"][best_epoch_idx],
                "best_val_average_angle_between_points": history["val_average_angle_between_points"][best_epoch_idx],
                "best_val_minimum_angle_between_points": history["val_minimum_angle_between_points"][best_epoch_idx],

                "best_val_configuration_rectification_mae": history["val_configuration_rectification_mae"][best_epoch_idx],
                "best_val_end_effector_position_rectification_mae": history["val_end_effector_position_rectification_mae"][best_epoch_idx],
                "best_val_end_effector_rotation_rectification_mae": history["val_end_effector_rotation_rectification_mae"][
                    best_epoch_idx],
                "best_val_magnet_rectification_mae": history["val_magnet_rectification_mae"][best_epoch_idx],
                "best_val_goal_obj6D_position_rectification_mae": history["val_goal_obj6D_position_rectification_mae"][best_epoch_idx],
                "best_val_goal_obj6D_rotation_rectification_mae": history["val_goal_obj6D_rotation_rectification_mae"][best_epoch_idx],
                "best_val_obstacle6D"
                "_position_rectification_mae": history["val_obstacle6D"
                                                       "_position_rectification_mae"][best_epoch_idx],
                "best_val_obstacle6D"
                "_rotation_rectification_mae": history["val_obstacle6D"
                                                       "_rotation_rectification_mae"][best_epoch_idx],

                "best_val_configuration_rectification_l2": history["val_configuration_rectification_l2"][best_epoch_idx],
                "best_val_end_effector_position_rectification_l2": history["val_end_effector_position_rectification_l2"][best_epoch_idx],
                "best_val_end_effector_rotation_rectification_l2": history["val_end_effector_rotation_rectification_l2"][best_epoch_idx],
                "best_val_magnet_rectification_l2": history["val_magnet_rectification_l2"][best_epoch_idx],
                "best_val_goal_obj6D_position_rectification_l2": history["val_goal_obj6D_position_rectification_l2"][best_epoch_idx],
                "best_val_goal_obj6D_rotation_rectification_l2": history["val_goal_obj6D_rotation_rectification_l2"][best_epoch_idx],
                "best_val_obstacle6D"
                "_position_rectification_l2": history["val_obstacle6D"
                                                      "_position_rectification_l2"][best_epoch_idx],
                "best_val_obstacle6D"
                "_rotation_rectification_l2": history["val_obstacle6D"
                                                      "_rotation_rectification_l2"][best_epoch_idx],

                "best_val_tail_mean_step_size": history["val_tail_mean_step_size"][best_epoch_idx],
                "best_val_tail_fraction_static": history["val_tail_fraction_static"][best_epoch_idx],
                "best_val_estimated_effective_length": history["val_estimated_effective_length"][best_epoch_idx],

                "checkpoint_path": checkpoint_path,
            }

            file_exists = os.path.exists(Path(results_dir) / f"trajectory_model_sweep_runs.csv")
            keys = load_existing_keys(str(Path(results_dir) / f"trajectory_model_sweep_runs.csv"))
            if (run_id, cfg.num_epochs) in keys:
                continue

            with open(Path(results_dir) / f"trajectory_model_sweep_runs.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

            if after_run_callback is not None:
                after_run_callback(Path(results_dir) / f"trajectory_model_sweep_runs.csv")

            logger.info("Finished run=%03d best_val_loss=%.6f.", run_id, row["best_val_loss"])

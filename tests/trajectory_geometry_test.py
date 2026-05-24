from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from remake.models.trajectory_model.trajectory_model import TrajectoryModel
from remake.models.create_datasets import PretrainTrajectoryModelDataset
from remake.models.trajectory_model.trajectory_model_training import move_batch_to_device, compute_tail_metrics, compute_angle_metrics


@torch.no_grad()
def compute_path_quality_metrics(
    predicted_trajectory: Dict[str, torch.Tensor],
    initial_state: Dict[str, torch.Tensor],
    final_state: Dict[str, torch.Tensor],
) -> Dict[str, float]:

    ee_pos = predicted_trajectory["end_effector_position"]

    initial_pos = initial_state["end_effector"][:, :3]
    final_pos = final_state["end_effector"][:, :3]

    metrics = {}

    metrics["distance_to_initial_position"] = torch.norm(
        initial_pos - ee_pos[:, 0, :],
        dim=-1,
    ).mean().item()

    metrics["distance_to_goal_position"] = torch.norm(
        final_pos - ee_pos[:, -1, :],
        dim=-1,
    ).mean().item()

    points = torch.cat(
        [
            initial_pos.unsqueeze(1),
            ee_pos,
            final_pos.unsqueeze(1),
        ],
        dim=1,
    )

    segment_lengths = torch.norm(
        points[:, 1:, :] - points[:, :-1, :],
        dim=-1,
    )

    average_spacing = segment_lengths.mean(dim=-1)

    spacing_deviation = torch.abs(
        segment_lengths - average_spacing.unsqueeze(-1)
    ).max(dim=-1).values

    metrics["average_spacing_between_points"] = average_spacing.mean().item()
    metrics["max_spacing_deviation"] = spacing_deviation.mean().item()

    metrics.update(
        compute_angle_metrics(
            ee_pos,
            initial_pos,
            final_pos,
        )
    )

    metrics.update(
        compute_tail_metrics(
            ee_pos,
        )
    )

    return metrics


@torch.no_grad()
def evaluate_trajectory_model(
    model,
    dataloader,
    device,
) -> Dict[str, Dict[str, float]]:

    model.eval()

    metric_values: Dict[str, list[float]] = {}

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        initial_state = {
            "configuration": batch["initial_state_configuration"],
            "end_effector": batch["initial_state_end_effector"],
            "magnet": batch["initial_state_magnet"],
            "goal_obj6D": batch["initial_state_goal_obj6D"],
            "obstacle6D": batch["initial_state_obstacle6D"],
        }

        final_state = {
            "configuration": batch["final_state_configuration"],
            "end_effector": batch["final_state_end_effector"],
            "magnet": batch["final_state_magnet"],
            "goal_obj6D": batch["final_state_goal_obj6D"],
            "obstacle6D": batch["final_state_obstacle6D"],
        }

        predicted_trajectory = model(x)

        metrics = compute_path_quality_metrics(
            predicted_trajectory,
            initial_state,
            final_state,
        )

        for key, value in metrics.items():
            if key not in metric_values:
                metric_values[key] = []

            metric_values[key].append(value)

    results: Dict[str, Dict[str, float]] = {}

    for key, values in metric_values.items():
        results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return results


def build_obstacle_split_datasets(
    h5_path: str,
    master_index_path: str,
    seed: int,
    max_ep_len: int = 52,
):
    df = pd.read_csv(master_index_path)

    df = df[df["ep_len"] <= max_ep_len]
    df = df[df["collision"] == 0]
    df.sort_values(by="episode_id", inplace=True)

    df = df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    no_obstacle_df = df[df["obstacle_active"] == 0].copy()
    obstacle_df = df[df["obstacle_active"] == 1].copy()

    no_obstacle_ids = no_obstacle_df["episode_id"].tolist()
    obstacle_ids = obstacle_df["episode_id"].tolist()

    return {
        "No obstacle": PretrainTrajectoryModelDataset(
            h5_path,
            no_obstacle_ids,
        ),
        "Obstacle": PretrainTrajectoryModelDataset(
            h5_path,
            obstacle_ids,
        ),
    }


def load_trajectory_model(
    checkpoint_path: str,
    device: str,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    cfg = checkpoint["config"]

    model = TrajectoryModel(
        input_dimension=58,
        n_gru=cfg["n_gru"],
        dimension_gru=cfg["d_gru"],
        output_layers=cfg["output_layers"],
        device=device,
        hidden_head_dimension=cfg["hidden_head_dimension"],
        n_timesteps=cfg["n_timesteps"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def add_composite_score(
    input_df: pd.DataFrame,
) -> pd.DataFrame:

    df = pd.DataFrame(input_df)

    score_columns = [
        "distance_to_goal_position_mean",
        "distance_to_initial_position_mean",
        "average_spacing_between_points_mean",
        "max_spacing_deviation_mean",
        "tail_mean_step_size_mean",
        "tail_fraction_static_mean",
        "estimated_effective_length_mean",
        "average_angle_between_points_mean",
    ]

    for col in score_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["composite_score"] = (
        df["distance_to_goal_position_mean"]
        + df["distance_to_initial_position_mean"]
        + df["average_spacing_between_points_mean"]
        + df["max_spacing_deviation_mean"]
        + df["tail_mean_step_size_mean"]
        + df["tail_fraction_static_mean"]
        + df["estimated_effective_length_mean"] / 50.0
        + (180.0 - df["average_angle_between_points_mean"]) / 180.0
    )

    return df


def format_mean_std(
    mean: float,
    std: float,
) -> str:
    return f"{mean:.6f} ± {std:.6f}"


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = Path(
        "dataset/test_trajectory"
    )

    h5_path = str(DATA_DIR / "worker_1.h5")
    master_index_path = str(DATA_DIR / "test_trajectory_master_index.csv")

    output_dir = Path("plots/trajectory_quality")
    output_dir.mkdir(parents=True, exist_ok=True)

    MODEL_RUNS = {
        "TM1": "final/small",
        "TM2": "final/large",
        "TM3": "final/priors",
        "TM4": "final/pretraining",
        "TM5": "final/supervised",
    }

    EPOCHS = [10, 20, 30, 40, 50]
    SEEDS = [1, 2, 3, 4, 5]

    all_rows: list[Dict[str, Any]] = []

    for model_name, run_dir in MODEL_RUNS.items():

        for epoch in EPOCHS:

            checkpoint_path = Path(run_dir) / f"plots/model_{epoch:04d}.pt"

            if not checkpoint_path.exists():
                print(f"Missing checkpoint: {checkpoint_path}")
                continue

            print(f"Evaluating {model_name}, epoch {epoch}")

            model = load_trajectory_model(
                checkpoint_path=str(checkpoint_path),
                device=device,
            )

            for seed in SEEDS:

                print(f"  Seed {seed}")

                datasets = build_obstacle_split_datasets(
                    h5_path=h5_path,
                    master_index_path=master_index_path,
                    seed=seed,
                )

                dataloaders = {
                    category: DataLoader(
                        dataset,
                        batch_size=64,
                        shuffle=False,
                        num_workers=10,
                    )
                    for category, dataset in datasets.items()
                }

                for category, dataloader in dataloaders.items():

                    results = evaluate_trajectory_model(
                        model=model,
                        dataloader=dataloader,
                        device=device,
                    )

                    row: Dict[str, Any] = {
                        "model": model_name,
                        "epoch": epoch,
                        "seed": seed,
                        "category": category,
                    }

                    for metric_name, stats in results.items():
                        row[f"{metric_name}_mean"] = stats["mean"]
                        row[f"{metric_name}_std"] = stats["std"]

                    all_rows.append(row)

    raw_df = pd.DataFrame(all_rows)

    raw_csv_path = output_dir / "trajectory_quality_raw_seed_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    metric_mean_columns = [
        col for col in raw_df.columns
        if col.endswith("_mean")
    ]

    grouped = raw_df.groupby(
        ["model", "epoch", "category"],
        as_index=False,
    )

    agg_rows: list[Dict[str, Any]] = []

    for (model_name, epoch, category), group in grouped:

        row: Dict[str, Any] = {
            "model": model_name,
            "epoch": int(epoch),
            "category": category,
        }

        for col in metric_mean_columns:
            values = group[col].to_numpy(dtype=float)

            metric_name = col.removesuffix("_mean")

            mean = float(np.mean(values))
            std = float(np.std(values))

            row[metric_name] = format_mean_std(mean, std)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std

        agg_rows.append(row)

    full_df = pd.DataFrame(agg_rows)

    full_csv_path = output_dir / "trajectory_quality_all_epochs.csv"
    full_latex_path = output_dir / "trajectory_quality_all_epochs.tex"

    full_df.to_csv(full_csv_path, index=False)

    full_df.drop(
        columns=[
            c for c in full_df.columns
            if c.endswith("_mean") or c.endswith("_std")
        ],
        errors="ignore",
    ).to_latex(
        full_latex_path,
        index=False,
        escape=False,
    )

    scored_df = add_composite_score(full_df)

    best_rows: list[Dict[str, Any]] = []

    for model_name in MODEL_RUNS.keys():

        model_df = scored_df[scored_df["model"] == model_name].copy()

        if len(model_df) == 0:
            continue

        epoch_scores = (
            model_df
            .groupby("epoch")["composite_score"]
            .mean()
            .reset_index()
        )

        best_epoch = int(
            epoch_scores
            .sort_values("composite_score")
            .iloc[0]["epoch"]
        )

        best_epoch_df = model_df[model_df["epoch"] == best_epoch].copy()

        for _, row in best_epoch_df.iterrows():
            best_rows.append(dict(row))

    best_df = pd.DataFrame(best_rows)

    best_csv_path = output_dir / "trajectory_quality_best_epochs.csv"
    best_latex_path = output_dir / "trajectory_quality_best_epochs.tex"

    best_df.to_csv(best_csv_path, index=False)

    best_df.drop(
        columns=[
            c for c in best_df.columns
            if c.endswith("_mean") or c.endswith("_std")
        ],
        errors="ignore",
    ).to_latex(
        best_latex_path,
        index=False,
        escape=False,
    )
from pathlib import Path
from typing import Dict
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from remake.models.create_datasets import internal_test_sets
from remake.models.forward_model.forward_model import ForwardModel
from remake.models.inverse_model.inverse_model import InverseModel
from remake.models.trajectory_model.trajectory_model import TrajectoryModel
from remake.models.create_datasets import PretrainTrajectoryModelDataset


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def measure_inference_time(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    n_warmup_batches: int = 10,
) -> Dict[str, float]:
    model.eval()

    sample_times_ms = []

    batch_idx = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        x = batch["input"]

        if batch_idx < n_warmup_batches:
            _ = model(x)
            batch_idx += 1
            continue

        if "cuda" in device:
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(x)

        if "cuda" in device:
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed = end - start
        batch_size = x.shape[0]

        sample_times_ms.append((elapsed / batch_size) * 1000.0)

    return {
        "mean_ms": float(np.mean(sample_times_ms)),
        "std_ms": float(np.std(sample_times_ms)),
    }


def load_model(
    model_name: str,
    model_info: Dict[str, str],
    device: str,
):
    checkpoint = torch.load(
        model_info["path"],
        map_location=device,
    )

    cfg = checkpoint["config"]

    if model_info["type"] == "inverse":
        model = InverseModel(
            input_dimension=50,
            hidden_dimension=cfg["hidden_dimension"],
            output_dimension=cfg["output_dimension"],
            n_hidden_layer=cfg["n_hidden_layers"],
        )

    elif model_info["type"] == "forward":
        model = ForwardModel(
            input_dimension=37,
            shared_hidden_dimension=cfg["shared_hidden_dimension"],
            n_shared_hidden_layers=cfg["n_shared_hidden_layers"],
            head_hidden_dimension=cfg["head_hidden_dimension"],
            output_layers=cfg["output_layers"],
            dropout_rate=cfg["dropout_rate"],
        )
    elif model_info["type"] == "trajectory":
        model = TrajectoryModel(
            input_dimension=58,
            n_gru=cfg["n_gru"],
            dimension_gru=cfg["d_gru"],
            output_layers=cfg["output_layers"],
            device=device,
            hidden_head_dimension=cfg["hidden_head_dimension"],
            n_timesteps=cfg["n_timesteps"],
        )

    else:
        raise ValueError(f"Unknown model type for {model_name}: {model_info['type']}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_test_datasets(
    model_type: str,
    master_index_path: str,
    h5_path: str,
):
    if model_type == "forward" or model_type == "inverse":
        return internal_test_sets(
            master_index_path,
            h5_path,
            mode_name=model_type
        )

    if model_type == "trajectory":

        df = pd.read_csv(master_index_path)

        df = df[df["collision"] == 0]
        df = df[df["ep_len"] <= 52]

        no_obstacle_df = df[
            df["obstacle_active"] == 0
        ].copy()

        obstacle_df = df[
            df["obstacle_active"] == 1
        ].copy()

        no_obstacle_dataset = PretrainTrajectoryModelDataset(
            h5_path,
            no_obstacle_df["episode_id"].tolist(),
        )

        obstacle_dataset = PretrainTrajectoryModelDataset(
            h5_path,
            obstacle_df["episode_id"].tolist(),
        )

        return (
            no_obstacle_dataset,
            obstacle_dataset,
            obstacle_dataset,
        )

    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":

    device = "cuda"

    MODEL_PATHS = {
        "IM1": {
            "type": "inverse",
            "path": "inverse.pt",
        },
        "IM2": {
            "type": "inverse",
            "path": "inverse2.pt",
        },
        "FM1": {
            "type": "forward",
            "path": "forward.pt",
        },
        "FM2": {
            "type": "forward",
            "path": "forward2.pt",
        },

        "TM1": {
            "type": "trajectory",
            "path": "model_0050.pt",
        },
        "TM2": {
            "type": "trajectory",
            "path": "model_0050.pt",
        },
        "TM3": {
            "type": "trajectory",
            "path": "model_0050.pt",
        },
        "TM4": {
            "type": "trajectory",
            "path": "model_0050.pt",
        },
        "TM5": {
            "type": "trajectory",
            "path": "model_0050.pt",
        },
    }

    babbling_master_index_path = (
        "dataset/test_babbling/test_babbling_master_index.csv"
    )

    babbling_h5_path = (
        "dataset/test_babbling/worker_1.h5"
    )

    trajectory_master_index_path = (
        "dataset/test_trajectory/test_trajectory_master_index.csv"
    )

    trajectory_h5_path = (
        "dataset/test_trajectory/worker_1.h5"
    )

    output_dir = Path("plots/inference_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    timing_rows = []

    for model_name, model_info in MODEL_PATHS.items():

        print(f"Evaluating inference time for {model_name}")

        model = load_model(
            model_name=model_name,
            model_info=model_info,
            device=device,
        )

        if model_info["type"] == "trajectory":

            current_master_index_path = trajectory_master_index_path
            current_h5_path = trajectory_h5_path

        else:

            current_master_index_path = babbling_master_index_path
            current_h5_path = babbling_h5_path

        no_obstacle_dataset, obstacle_collision_dataset, obstacle_avoidance_dataset = load_test_datasets(
            model_type=model_info["type"],
            master_index_path=current_master_index_path,
            h5_path=current_h5_path,
        )

        no_obstacle_loader = DataLoader(
            no_obstacle_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        obstacle_dataset = ConcatDataset(
            [
                obstacle_collision_dataset,
                obstacle_avoidance_dataset,
            ]
        )

        obstacle_loader = DataLoader(
            obstacle_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        no_obstacle_time = measure_inference_time(
            model=model,
            dataloader=no_obstacle_loader,
            device=device,
            n_warmup_batches=10,
        )

        obstacle_time = measure_inference_time(
            model=model,
            dataloader=obstacle_loader,
            device=device,
            n_warmup_batches=10,
        )

        row = {
            "model": model_name,
            "model_type": model_info["type"],
            "no_obstacle_ms": (
                f"{no_obstacle_time['mean_ms']:.6f} ± "
                f"{no_obstacle_time['std_ms']:.6f}"
            ),
            "obstacle_ms": (
                f"{obstacle_time['mean_ms']:.6f} ± "
                f"{obstacle_time['std_ms']:.6f}"
            ),
        }

        timing_rows.append(row)

    timing_df = pd.DataFrame(timing_rows)

    csv_path = output_dir / "inference_time_results.csv"
    latex_path = output_dir / "inference_time_results.tex"

    timing_df.to_csv(
        csv_path,
        index=False,
    )

    timing_df.to_latex(
        latex_path,
        index=False,
        escape=False,
    )
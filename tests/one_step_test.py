from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.create_datasets import internal_test_sets
from models.forward_model.forward_model import ForwardModel
from models.inverse_model.inverse_model import InverseModel
from models.forward_model.loss_function import quaternion_geodesic_loss


def move_batch_to_device(
        batch: Dict[str, torch.Tensor],
        device: str
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}

def compute_inverse_metrics_per_category(
        dataloader: DataLoader,
        model: InverseModel,
        device: str,
):
    total_q_sample_mae = 0.0
    total_mgt_sample_mae = 0.0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            x = batch["input"]
            target = batch["action"]

            preds = model(x)

            q_sample_mae = torch.abs(preds[:, :-1] - target[:, :-1]).mean(dim=1)
            mgt_sample_mae = torch.abs(preds[:, -1] - target[:, -1])

            total_q_sample_mae += q_sample_mae.sum().item()
            total_mgt_sample_mae += mgt_sample_mae.sum().item()
            total_samples += target.shape[0]

    return {
        r'$\Delta \theta$ MAE': total_q_sample_mae / total_samples,
        r'$\Delta mgt$ error': total_mgt_sample_mae / total_samples,
    }

def compute_forward_metrics_per_category(
        dataloader: DataLoader,
        model: ForwardModel,
        device: str,
):
    total_q_sample_mae = 0.0
    total_ef_pos_sample_mae = 0.0
    total_ef_rot_sample_mae = 0.0
    total_mgt_sample_mae = 0.0
    total_goal_pos_sample_mae = 0.0
    total_goal_rot_sample_mae = 0.0
    total_obs_pos_sample_mae = 0.0
    total_obs_rot_sample_mae = 0.0

    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            x = batch["input"]
            target = {
                "configuration": batch["next_configuration"],
                "end_effector": batch["next_end_effector"],
                "magnet": batch["next_magnet"],
                "goal_obj6D": batch["next_goal_obj6D"],
                "occ6D": batch["next_occ6D"],
            }

            preds = model(x)

            q_sample_mae = torch.abs(preds["configuration"] - target["configuration"]).mean(dim=1)
            ef_pos_sample_mae = torch.abs(preds["end_effector_position"] - target["end_effector"][:, :3]).mean(dim=1)
            ef_rot_sample_mae = quaternion_geodesic_loss(preds["end_effector_rotation"], target["end_effector"][:, 3:])
            mgt_sample_mae = torch.abs(preds["magnet"] - target["magnet"]).mean(dim=1)
            goal_pos_sample_mae = torch.abs(preds["goal_obj6D_position"] - target["goal_obj6D"][:, :3]).mean(dim=1)
            goal_rot_sample_mae = quaternion_geodesic_loss(preds["goal_obj6D_rotation"], target["goal_obj6D"][:, 3:])
            obs_pos_sample_mae = torch.abs(preds["occ6D_position"] - target["occ6D"][:, :3]).mean(dim=1)
            obs_rot_sample_mae = quaternion_geodesic_loss(preds["occ6D_rotation"], target["occ6D"][:, 3:])

            total_q_sample_mae += q_sample_mae.sum().item()
            total_ef_pos_sample_mae += ef_pos_sample_mae.sum().item()
            total_ef_rot_sample_mae += ef_rot_sample_mae.mean().item() * target["configuration"].shape[0]
            total_mgt_sample_mae += mgt_sample_mae.sum().item()
            total_goal_pos_sample_mae += goal_pos_sample_mae.sum().item()
            total_goal_rot_sample_mae += goal_rot_sample_mae.mean().item() * target["configuration"].shape[0]
            total_obs_pos_sample_mae += obs_pos_sample_mae.sum().item()
            total_obs_rot_sample_mae += obs_rot_sample_mae.mean().item() * target["configuration"].shape[0]
            total_samples += target["configuration"].shape[0]

    return {
        r'$\mathbf{\theta}$ MAE[rad]': total_q_sample_mae / total_samples,
        r'$\mathbf{ef}_{xyz}$ MAE[m]': total_ef_pos_sample_mae / total_samples,
        r'$\mathbf{ef}_{R}$ error[rad]': total_ef_rot_sample_mae / total_samples,
        r'$mgt$ MAE': total_mgt_sample_mae / total_samples,
        r'$\mathbf{g}_{xyz}$ MAE[m]': total_goal_pos_sample_mae / total_samples,
        r'$\mathbf{g}_{R}$ error[rad]': total_goal_rot_sample_mae / total_samples,
        r'$\mathbf{o}_{xyz}$ MAE[m]': total_obs_pos_sample_mae / total_samples,
        r'$\mathbf{o}_{R}$ error[rad]': total_obs_rot_sample_mae / total_samples,
    }

def compute_metrics_per_category(
        dataloader: DataLoader,
        model,
        device: str,
):
    if isinstance(model, InverseModel):
        return compute_inverse_metrics_per_category(dataloader, model, device)
    if isinstance(model, ForwardModel):
        return compute_forward_metrics_per_category(dataloader, model, device)
    else:
        return {}

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
}

PLOT_DIR = Path("plots/one_step")

H5_PATH = "dataset/test_babbling/worker_1.h5"
MASTER_INDEX_PATH = "dataset/test_babbling/test_babbling_master_index.csv"

if __name__ == "__main__":

    categories = ["No obstacle", "Colliding", "Non-colliding"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    inverse_table_rows = []
    forward_table_rows = []

    for model_name, model_info in MODEL_PATHS.items():

        print(f"Evaluating {model_name}")

        if model_info["type"] == "inverse":
            datasets = internal_test_sets(
                MASTER_INDEX_PATH,
                H5_PATH,
                "inverse"
            )

            checkpoint = torch.load(model_info["path"], map_location=device)
            cfg = checkpoint["config"]

            model = InverseModel(
                input_dimension=50,
                hidden_dimension=cfg["hidden_dimension"],
                output_dimension=cfg["output_dimension"],
                n_hidden_layer=cfg["n_hidden_layers"],
            )

        else:
            datasets = internal_test_sets(
                MASTER_INDEX_PATH,
                H5_PATH,
                "forward"
            )

            checkpoint = torch.load(model_info["path"], map_location=device)
            cfg = checkpoint["config"]

            model = ForwardModel(
                input_dimension=37,
                shared_hidden_dimension=cfg["shared_hidden_dimension"],
                n_shared_hidden_layers=cfg["n_shared_hidden_layers"],
                head_hidden_dimension=cfg["head_hidden_dimension"],
                output_layers=cfg["output_layers"],
                dropout_rate=cfg["dropout_rate"],
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        dataloaders = [
            DataLoader(ds, batch_size=128, shuffle=False, num_workers=10)
            for ds in datasets
        ]

        row = {"model": model_name}

        category_results = {}

        for category, dataloader in zip(categories, dataloaders):
            result = compute_metrics_per_category(dataloader, model, device)

            for metric_name, value in result.items():
                if metric_name not in category_results:
                    category_results[metric_name] = []

                category_results[metric_name].append(value)

                row[f"{category} {metric_name}"] = f"{value:.6f}"

        for metric_name, values in category_results.items():
            row[f"mean {metric_name}"] = f"{np.mean(values):.6f}"
            row[f"std {metric_name}"] = f"{np.std(values):.6f}"

        if model_info["type"] == "inverse":
            inverse_table_rows.append(row)
        else:
            forward_table_rows.append(row)

        for metric_name, values in category_results.items():
            x = np.arange(len(categories))

            plt.figure(figsize=(6, 5))

            plt.bar(x, values)

            plt.xticks(x, categories, rotation=20, ha="right")
            plt.ylabel(metric_name)
            plt.title(f"{model_name}: {metric_name}")

            plt.grid(True, axis="y")
            for i, v in enumerate(values):
                plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout(pad=1.0)

            safe_metric_name = (
                metric_name
                .replace("$", "")
                .replace("\\", "")
                .replace("{", "")
                .replace("}", "")
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "_")
                .replace("/", "_")
            )

            plt.savefig(
                PLOT_DIR / f"{model_name}_{safe_metric_name}_one_step.pdf"
            )

            plt.close()

    inverse_df = pd.DataFrame(inverse_table_rows)
    inverse_df = inverse_df.sort_values("model")

    inverse_csv_path = PLOT_DIR / "inverse_one_step_results.csv"
    inverse_latex_path = PLOT_DIR / "inverse_one_step_results.tex"

    inverse_df.to_csv(inverse_csv_path, index=False)

    inverse_df.to_latex(
        inverse_latex_path,
        index=False,
        escape=False,
    )

    forward_df = pd.DataFrame(forward_table_rows)
    forward_df = forward_df.sort_values("model")

    forward_csv_path = PLOT_DIR / "forward_one_step_results.csv"
    forward_latex_path = PLOT_DIR / "forward_one_step_results.tex"

    forward_df.to_csv(forward_csv_path, index=False)

    forward_df.to_latex(
        forward_latex_path,
        index=False,
        escape=False,
    )

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from models.forward_model.forward_model import ForwardModel
from models.create_datasets import trajectory_sweep_dataset
from models.inverse_model.inverse_model import InverseModel
from models.trajectory_model.trajectory_model_training import move_batch_to_device, rectify_trajectory, compute_rectification_mae


@torch.no_grad()
def test_rectification_error(
    forward_model,
    inverse_model,
    dataloader,
    device,
):
    forward_model.eval()
    inverse_model.eval()

    metric_values = {}

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        initial_state = {
            "configuration": batch["initial_state_configuration"],
            "end_effector": batch["initial_state_end_effector"],
            "magnet": batch["initial_state_magnet"],
            "goal_obj6D": batch["initial_state_goal_obj6D"],
            "obstacle6D": batch["initial_state_obstacle6D"],
        }

        predicted_trajectory = {
            "configuration": batch["trajectory_configuration"],
            "end_effector_position": batch["trajectory_end_effector_position"],
            "end_effector_rotation": batch["trajectory_end_effector_rotation"],
            "magnet": batch["trajectory_magnet"],
            "goal_obj6D_position": batch["trajectory_goal_obj6D_position"],
            "goal_obj6D_rotation": batch["trajectory_goal_obj6D_rotation"],
            "obstacle6D_position": batch["trajectory_obstacle6D_position"],
            "obstacle6D_rotation": batch["trajectory_obstacle6D_rotation"],
        }

        rectified_trajectory = rectify_trajectory(
            forward_model,
            inverse_model,
            predicted_trajectory,
            initial_state,
        )

        metrics = compute_rectification_mae(
            predicted_trajectory,
            rectified_trajectory,
        )

        for key, value in metrics.items():
            if key not in metric_values:
                metric_values[key] = []

            metric_values[key].append(value)

    results = {}

    for key, values in metric_values.items():
        results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return results

# Also include which is inverse/forward
MODEL_PATHS = {
    "FM1 + IM1": {
        "forward": "forward.pt",
        "inverse": "inverse.pt",
    },
    "FM1 + IM2": {
        "forward": "forward.pt",
        "inverse": "inverse2.pt",
    },
    "FM2 + IM1": {
        "forward": "forward2.pt",
        "inverse": "inverse.pt",
    },
    "FM2 + IM2": {
        "forward": "forward2.pt",
        "inverse": "inverse2.pt",
    },
}

DATA_FILE = Path(
        "dataset/trajectory/trajectory_dataset.h5"
    )

DATA_IDX_FILE = Path(
        "dataset/trajectory/trajectory_master_index_file.csv"
    )

SEEDS = [1, 2, 3, 4, 5]

CSV_PATH = "plots/rectification/rectification_results.csv"
LATEX_PATH = "plots/rectification/rectification_results.tex"

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    name_dict = {
        "configuration_rectification_mae": r'$\mathbf{\theta}$ MAE[rad]',
        "end_effector_position_rectification_mae": r'$\mathbf{ef}_{xyz}$ MAE[m]',
        "end_effector_rotation_rectification_mae": r'$\mathbf{ef}_{R}$ MGE[rad]',
        "magnet_rectification_mae": r'$mgt$ MAE',
        "goal_obj6D_position_rectification_mae": r'$\mathbf{g}_{xyz}$ MAE[m]',
        "goal_obj6D_rotation_rectification_mae": r'$\mathbf{g}_{R}$ MGE[rad]',
        "obstacle6D_position_rectification_mae": r'$\mathbf{o}_{xyz}$ MAE[m]',
        "obstacle6D_rotation_rectification_mae": r'$\mathbf{o}_{R}$ MGE[rad]',
        "configuration_rectification_l2": r'$\mathbf{\theta}$ L2[rad]',
        "end_effector_position_rectification_l2": r'$\mathbf{ef}_{xyz}$ L2[m]',
        "end_effector_rotation_rectification_l2": r'$\mathbf{ef}_{R}$ L2',
        "magnet_rectification_l2": r'$mgt$ L2',
        "goal_obj6D_position_rectification_l2": r'$\mathbf{g}_{xyz}$ L2[m]',
        "goal_obj6D_rotation_rectification_l2": r'$\mathbf{g}_{R}$ L2',
        "obstacle6D_position_rectification_l2": r'$\mathbf{o}_{xyz}$ L2[m]',
        "obstacle6D_rotation_rectification_l2": r'$\mathbf{o}_{R}$ L2',
    }

    table_rows = []

    for model_name, model_paths in MODEL_PATHS.items():

        print(f"Evaluating {model_name}")

        checkpoint = torch.load(
            model_paths["inverse"],
            map_location=device,
        )

        im_cfg = checkpoint["config"]

        inverse_model = InverseModel(
            input_dimension=50,
            hidden_dimension=im_cfg["hidden_dimension"],
            output_dimension=im_cfg["output_dimension"],
            n_hidden_layer=im_cfg["n_hidden_layers"],
        )

        inverse_model.load_state_dict(checkpoint["model_state_dict"])
        inverse_model.to(device)
        inverse_model.eval()

        for param in inverse_model.parameters():
            param.requires_grad = False

        checkpoint = torch.load(
            model_paths["forward"],
            map_location=device,
        )

        fm_cfg = checkpoint["config"]

        forward_model = ForwardModel(
            input_dimension=37,
            shared_hidden_dimension=fm_cfg["shared_hidden_dimension"],
            n_shared_hidden_layers=fm_cfg["n_shared_hidden_layers"],
            head_hidden_dimension=fm_cfg["head_hidden_dimension"],
            output_layers=fm_cfg["output_layers"],
            dropout_rate=fm_cfg["dropout_rate"],
        )

        forward_model.load_state_dict(checkpoint["model_state_dict"])
        forward_model.to(device)
        forward_model.eval()

        for param in forward_model.parameters():
            param.requires_grad = False

        all_results = []

        for seed in SEEDS:
            train_dataset, val_dataset, _ = trajectory_sweep_dataset(
                h5_file_path=str(DATA_FILE),
                master_index_df_path=str(DATA_IDX_FILE),
                n_trajectories=13000,
                target_no_obstacle_ratio=0.1,
                seed=seed,
            )

            loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=10,
            )

            results = test_rectification_error(
                forward_model=forward_model,
                inverse_model=inverse_model,
                dataloader=loader,
                device=device,
            )

            all_results.append(results)

        metric_names = all_results[0].keys()

        row = {
            "model": model_name
        }

        for metric_name in metric_names:

            seed_values = []

            for result in all_results:
                seed_values.append(result[metric_name]["mean"])

            seed_values = np.asarray(seed_values)

            mean = seed_values.mean()
            std = seed_values.std()

            row[metric_name] = f"{mean:.6f} ± {std:.6f}"

        table_rows.append(row)

        plot_dir = Path("plots/rectification")
        plot_dir.mkdir(parents=True, exist_ok=True)

        safe_model_name = model_name.replace(" ", "_").replace("+", "plus")

        for metric_name in metric_names:

            seed_values = []

            for result in all_results:
                seed_values.append(result[metric_name]["mean"])

            seed_values = np.asarray(seed_values)

            mean = seed_values.mean()
            std = seed_values.std()

            plt.figure(figsize=(6, 5))

            plt.errorbar(
                [1],
                [mean],
                yerr=[std],
                fmt="o",
                capsize=6,
            )

            name = name_dict.get(metric_name, metric_name)

            plt.xticks([1], [model_name])
            plt.ylabel(name)
            plt.title(f"{name} rectification error")

            plt.tight_layout()

            plt.savefig(
                plot_dir / f"{safe_model_name}_{metric_name}_rectification.pdf"
            )

    df = pd.DataFrame(table_rows)

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    df.to_latex(
        LATEX_PATH,
        index=False,
        escape=False,
    )

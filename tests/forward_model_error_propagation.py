from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import h5py

from models.forward_model.forward_model import ForwardModel
from models.forward_model.forward_model_training import compute_forward_metrics


class ForwardModelRolloutDataset(Dataset):

    def __init__(
        self,
        h5_path: str,
        master_index_df_path: str,
        max_rollout_steps: int = 20,
        seed: int = 1,
        obstacle_active: int | None = None,
    ):
        self.h5_path = h5_path
        self.max_rollout_steps = max_rollout_steps
        self.h5_file = None


        rng = np.random.default_rng(seed)
        df = pd.read_csv(master_index_df_path)

        df = df[df["collision"] == 0].copy()
        if obstacle_active is not None:
            df = df[df["obstacle_active"] == obstacle_active].copy()

        rollout_specs: List[Tuple[int, int]] = []

        for setup_id in df["setup_id"].unique():
            setup_eps = df[df["setup_id"] == setup_id].copy()

            if len(setup_eps) == 0:
                continue

            episode = setup_eps.sample(
                n=1,
                random_state=int(rng.integers(0, 2**31 - 1))
            ).iloc[0]

            ep_start = int(episode["ep_start"])
            ep_len = int(episode["ep_len"])

            if ep_len - 1 < max_rollout_steps:
                continue

            rollout_len = max_rollout_steps

            if rollout_len <= 0:
                continue

            rollout_specs.append((ep_start, rollout_len))

        self.rollout_specs = rollout_specs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.h5_file = None

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.h5_file

    def __len__(self):
        return len(self.rollout_specs)

    @staticmethod
    def _fix_occ(occ: np.ndarray) -> np.ndarray:
        invalid = np.asarray(
            [-100, -100, -100, -100, -100, -100, -100],
            dtype=np.float64,
        )
        replacement = np.asarray(
            [-2, -2, -2, -2, -2, -2, -2],
            dtype=np.float64,
        )

        if np.array_equal(occ, invalid):
            return replacement

        return occ

    def _make_input(self, transitions, idx: int) -> torch.Tensor:
        obstacle6D_t = self._fix_occ(transitions["obstacle6D_t"][idx])

        x = np.concatenate(
            (
                transitions["joints_angles_t"][idx],
                transitions["ee6D_t"][idx],
                np.asarray([transitions["mgt_t"][idx]]),
                transitions["goal_obj6D_t"][idx],
                obstacle6D_t,
                transitions["desired_delta_q"][idx],
                np.asarray([transitions["delta_mgt"][idx]]),
            )
        )

        return torch.from_numpy(x).float()

    def _make_target(self, transitions, idx: int) -> Dict[str, torch.Tensor]:
        obstacle6D_t1 = self._fix_occ(transitions["obstacle6D_t1"][idx])

        return {
            "next_configuration": torch.from_numpy(
                transitions["joints_angles_t1"][idx]
            ).float(),
            "next_end_effector": torch.from_numpy(
                transitions["ee6D_t1"][idx]
            ).float(),
            "next_magnet": torch.from_numpy(
                np.asarray([transitions["mgt_t1"][idx]])
            ).float(),
            "next_goal_obj6D": torch.from_numpy(
                transitions["goal_obj6D_t1"][idx]
            ).float(),
            "next_obstacle6D": torch.from_numpy(
                obstacle6D_t1
            ).float(),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        ep_start, rollout_len = self.rollout_specs[idx]

        h5_file = self._get_h5_file()
        transitions = h5_file["transitions"]

        indices = list(range(ep_start, ep_start + rollout_len))

        inputs = []
        targets = {
            "next_configuration": [],
            "next_end_effector": [],
            "next_magnet": [],
            "next_goal_obj6D": [],
            "next_obstacle6D": [],
        }

        for transition_idx in indices:
            inputs.append(self._make_input(transitions, transition_idx))

            target = self._make_target(transitions, transition_idx)
            for key in targets:
                targets[key].append(target[key])

        inputs = torch.stack(inputs, dim=0)

        stacked_targets = {
            key: torch.stack(value, dim=0)
            for key, value in targets.items()
        }

        return {
            "input": inputs,
            "target": stacked_targets,
            "rollout_len": torch.tensor(rollout_len, dtype=torch.long),
            "ep_start": torch.tensor(ep_start, dtype=torch.long),
        }

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        self.close()

def test_rollout_error(
    model,
    dataloader,
    device,
):
    model.eval()

    per_step_errors = {}

    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch["input"].to(device)
            targets = {
                k: v.to(device)
                for k, v in batch["target"].items()
            }

            B, T, _ = x_seq.shape

            x_t = x_seq[:, 0, :]

            for t in range(T):
                prediction = model(x_t)

                target = {
                    "configuration": targets["next_configuration"][:, t, :],
                    "end_effector": targets["next_end_effector"][:, t, :],
                    "magnet": targets["next_magnet"][:, t, :],
                    "goal_obj6D": targets["next_goal_obj6D"][:, t, :],
                    "obstacle6D": targets["next_obstacle6D"][:, t, :],
                }

                errors = compute_forward_metrics(prediction, target)
                for key, v in errors.items():
                    if key not in per_step_errors:
                        per_step_errors[key] = [[] for _ in range(T)]

                    per_step_errors[key][t].append(v)

                if t < T - 1:
                    next_action_part = x_seq[:, t + 1, -8:]

                    x_t = torch.cat(
                        [
                            prediction["configuration"],
                            prediction["end_effector_position"],
                            prediction["end_effector_rotation"],
                            prediction["magnet"],
                            prediction["goal_obj6D_position"],
                            prediction["goal_obj6D_rotation"],
                            prediction["obstacle6D_position"],
                            prediction["obstacle6D_rotation"],
                            next_action_part,
                        ],
                        dim=1,
                    )

    results = {}

    for metric_name, step_errors in per_step_errors.items():

        means = []
        stds = []

        for errors_at_step in step_errors:

            if len(errors_at_step) == 0:
                means.append(float("nan"))
                stds.append(float("nan"))
            else:
                means.append(float(np.mean(errors_at_step)))
                stds.append(float(np.std(errors_at_step)))

        results[metric_name] = {
            "mean": means,
            "std": stds,
        }

    return results

if __name__ == "__main__":
    name_dict = {
        "configuration_mae" : r'$\mathbf{\theta}$ MAE[rad]',
        "end_effector_position_mae" : r'$\mathbf{ef}_{xyz}$ MAE[m]',
        "end_effector_rotation_mae" : r'$\mathbf{ef}_{R}$ error[rad]',
        "magnet_acc" : r'$mgt$ MAE',
        "goal_obj6D_position_mae" : r'$\mathbf{g}_{xyz}$ MAE[m]',
        "goal_obj6D_rotation_mae" : r'$\mathbf{g}_{R}$ error[rad]',
        "obstacle6D_position_mae" : r'$\mathbf{o}_{xyz}$ MAE[m]',
        "obstacle6D_rotation_mae": r'$\mathbf{o}_{R}$ error[rad]',
    }

    device = "cpu"

    MODEL_PATHS = {
        "FM1": "forward.pt",
        "FM2": "forward2.pt",
    }

    OBSTACLE_PRESENT = {
        1: "obstacle",
        0: "no_obstacle",
    }

    seeds = [1, 2, 3, 4, 5]

    plot_dir = Path("plots/rollout")
    plot_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []

    for model_name, model_path in MODEL_PATHS.items():

        print(f"Evaluating {model_name}")
        obstacle_active = None

        checkpoint = torch.load(model_path, map_location=device)
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

        for seed in seeds:
            rollout_dataset = ForwardModelRolloutDataset(
                h5_path="data/trajectory_dataset.h5",
                master_index_df_path="data/trajectory_master_index.csv",
                max_rollout_steps=20,
                seed=seed,
                obstacle_active=obstacle_active,
            )

            rollout_loader = DataLoader(
                rollout_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=10,
            )

            results = test_rollout_error(
                model=forward_model,
                dataloader=rollout_loader,
                device=device,
            )

            all_results.append(results)

        metric_names = all_results[0].keys()

        row = {
            "model": model_name,
            "obstacle_active": obstacle_active,
        }

        for metric_name in metric_names:
            if "l2" in metric_name:
                continue

            seed_means = []

            for result in all_results:
                seed_means.append(result[metric_name]["mean"])

            seed_means = np.asarray(seed_means)

            mean_per_step = seed_means.mean(axis=0)
            std_per_step = seed_means.std(axis=0)

            overall_mean = mean_per_step.mean()
            overall_std = mean_per_step.std()

            row[metric_name] = f"{overall_mean:.6f} ± {overall_std:.6f}"

            steps = np.arange(1, len(mean_per_step) + 1)

            plt.figure(figsize=(8, 5))

            plt.errorbar(
                steps,
                mean_per_step,
                yerr=std_per_step,
                fmt="-o",
                capsize=4,
            )

            plt.xticks(steps)

            name = name_dict.get(metric_name, metric_name)

            if obstacle_active is None:
                text = ""
            else:
                if obstacle_active:
                    text = " with obstacle present"
                else:
                    text = " with no obstacle present"

            plt.xlabel("Rollout step")
            plt.ylabel(name)
            plt.title(f"{model_name}{text}: {name} rollout error propagation")
            plt.grid(True, axis="y")
            plt.tight_layout()

            safe_model_name = model_name.replace(" ", "_").replace("+", "plus")

            plt.savefig(
                plot_dir / f"{safe_model_name}_both_{metric_name}_rollout.pdf"
            )

            plt.close()

        table_rows.append(row)

    df = pd.DataFrame(table_rows)

    csv_path = plot_dir / "rollout_results.csv"
    latex_path = plot_dir / "rollout_results.tex"

    df.to_csv(csv_path, index=False)

    df.to_latex(
        latex_path,
        index=False,
        escape=False,
    )

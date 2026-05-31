from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Dict

import h5py
import numpy as np
import pandas as pd

@dataclass
class EpisodeIndexRow:
    file_path: str
    file_name: str
    file_stem: str

    episode_id: int
    global_env_key: str
    env_id: int

    setup_id: str

    ep_start: int
    ep_len: int
    n_transitions: int

    obstacle_active: int
    obstacle_dummy: int

    goal_obj6D_json: str
    obstacle6D_json: str

    planner: Optional[int] = None
    mode: Optional[int] = None
    collision: Optional[int] = None
    success: Optional[int] = None


def _round_array_for_hash(
        x: np.ndarray,
        decimals: int = 6
) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64), decimals=decimals)


def make_setup_id(
        goal_obj6D: np.ndarray,
        obstacle6D: np.ndarray
) -> str:
    """
    Generates unique setup ID of the environment based on 6D poses of the goal object and the obstacle.

    :param goal_obj6D: 6D pose of the goal object.
    :param obstacle6D: 6D pose of the obstacle.
    :return: String representing unique setup ID.
    """
    goal_arr = _round_array_for_hash(goal_obj6D)
    obstacle_arr = _round_array_for_hash(obstacle6D)

    payload = {
        "goal_obj6D": goal_arr.tolist(),
        "obstacle6D": obstacle_arr.tolist(),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def is_dummy_obstacle(
    obstacle6D: np.ndarray,
    atol: float = 1e-6,
) -> bool:
    """
    Checks whether the obstacle pose is the token for no obstacle.

    :param obstacle6D: 6D pose of the obstacle.
    :return: True if obstacle is not present.
    """

    obstacle = np.asarray(obstacle6D, dtype=np.float64)
    return np.allclose(obstacle, np.asarray([-2, -2, -2, -2, -2, -2, -2], dtype=np.float64), atol=atol, rtol=0.0)

def build_master_episode_index(
    h5_paths: Iterable[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Builds master episode index table. Stores information about the environment setups and episodes and indexes their
    corresponding transitions.

    :param h5_paths: File path of the data.
    :param output_path: Output file path for the table.
    :return: Dataframe of the created master index table.
    """
    rows: list[EpisodeIndexRow] = []

    for path_like in h5_paths:
        path = Path(path_like)

        with h5py.File(path, "r") as f:
            env_group = f["env"]
            ep_group = f["episodes"]

            env_goal = env_group["goal_obj6D"]
            env_obstacle = env_group["obstacle6D"]

            ep_env_id = ep_group["env_id"]
            ep_start = ep_group["ep_start"]
            ep_len = ep_group["ep_len"]

            planner_ds = ep_group["planner"] if "planner" in ep_group else None
            mode_ds = ep_group["mode"] if "mode" in ep_group else None
            collision_ds = ep_group["collision"] if "collision" in ep_group else None
            success_ds = ep_group["success"] if "success" in ep_group else None

            n_episodes = ep_env_id.shape[0]

            for episode_id in range(n_episodes):
                env_id = int(ep_env_id[episode_id])
                start = int(ep_start[episode_id])
                length = int(ep_len[episode_id])

                goal_obj6D = np.asarray(env_goal[env_id], dtype=np.float64)
                obstacle6D = np.asarray(env_obstacle[env_id], dtype=np.float64)

                setup_id = make_setup_id(goal_obj6D, obstacle6D)
                obstacle_dummy = int(is_dummy_obstacle(obstacle6D))
                obstacle_active = int(not obstacle_dummy)

                row = EpisodeIndexRow(
                    file_path=str(path.resolve()),
                    file_name=path.name,
                    file_stem=path.stem,

                    episode_id=episode_id,
                    global_env_key=f"{path.name}__env_{env_id}",
                    env_id=env_id,

                    setup_id=setup_id,

                    ep_start=start,
                    ep_len=length,
                    n_transitions=max(length, 0),

                    obstacle_active=obstacle_active,
                    obstacle_dummy=obstacle_dummy,

                    goal_obj6D_json=json.dumps(goal_obj6D.tolist()),
                    obstacle6D_json=json.dumps(obstacle6D.tolist()),

                    planner=int(planner_ds[episode_id]) if planner_ds is not None else None,
                    mode=int(mode_ds[episode_id]) if mode_ds is not None else None,
                    collision=int(collision_ds[episode_id]) if collision_ds is not None else None,
                    success=int(success_ds[episode_id]) if success_ds is not None else None,
                )
                rows.append(row)

    df = pd.DataFrame(asdict(r) for r in rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    return df

def compute_trajectory_stats(
    h5_file_path: str,
    master_index_df_path: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculates geometric properties of the generated trajectories in the training dataset. Metrics calculated are average
    step sizes and their deviations, and angles between consecutive waypoints and their deviations. Can be also used
    to calculate these metrics for transition dataset.

    :param h5_file_path: File path of the dataset.
    :param master_index_df_path: File path of the master index file.
    :return: Measured metrics.
    """
    df =master_index_df_path

    # same filtering as dataset
    #df = df[df["ep_len"] <= 52]
    #df= df[df["collision"] == 0]

    avg_spacings = []
    max_deviations = []

    avg_angles = []
    min_angles = []

    with h5py.File(h5_file_path, "r") as f:
        episodes = f["episodes"]
        transitions = f["transitions"]

        for _, row in df.iterrows():
            ep_id = int(row["episode_id"])

            start = int(episodes["ep_start"][ep_id])
            length = int(episodes["ep_len"][ep_id])

            ee_positions = []
            for t in range(length):
                idx = start + t
                ee6d = transitions["ee6D_t"][idx]
                ee_positions.append(ee6d[:3])

            ee_positions = np.array(ee_positions)

            deltas = np.linalg.norm(
                ee_positions[1:] - ee_positions[:-1],
                axis=-1
            )

            avg_spacing = np.mean(deltas)
            deviation = np.max(np.abs(deltas - avg_spacing))

            avg_spacings.append(avg_spacing)
            max_deviations.append(deviation)
            if len(ee_positions) < 3:
                continue

            v1 = ee_positions[1:-1] - ee_positions[:-2]
            v2 = ee_positions[2:] - ee_positions[1:-1]

            v1_norm = np.linalg.norm(v1, axis=-1)
            v2_norm = np.linalg.norm(v2, axis=-1)

            eps = 1e-8
            denom = np.maximum(v1_norm * v2_norm, eps)

            cos_theta = np.sum(v1 * v2, axis=-1) / denom
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angles_rad = np.arccos(cos_theta)
            angles_deg = angles_rad * (180.0 / np.pi)

            avg_angles.append(np.mean(angles_deg))
            min_angles.append(np.min(angles_deg))

    results = {
        "mean_average_spacing": float(np.mean(avg_spacings)),
        "std_average_spacing": float(np.std(avg_spacings)),

        "mean_max_deviation": float(np.mean(max_deviations)),
        "std_max_deviation": float(np.std(max_deviations)),

        "mean_average_angle": float(np.mean(avg_angles)),
        "std_average_angle": float(np.std(avg_angles)),

        "mean_minimum_angle": float(np.mean(min_angles)),
        "std_minimum_angle": float(np.std(min_angles)),
    }
    print(np.max(avg_spacings))

    return results

if __name__ == '__main__':
    mode = "trajectory"
    path = Path(f"dataset/test_{mode}/worker_1.h5")
    paths = [path]

    df = build_master_episode_index(
        h5_paths=paths,
        output_path=f"dataset/test_{mode}/test_{mode}_master_index.csv",
    )

    #p = pd.read_csv(f"dataset/{mode}2.0/{mode}_master_index.csv")
    #p = p[p["ep_len"] >= 20]
    #p = p[p["ep_len"] <= 52]
    #p = p[p["collision"] == 0]
    #print(len(p))
    #print(p["ep_len"].max())
    #a = df[df["obstacle_active"] == 0]
    #b = df[df["obstacle_active"] == 1]
    #c = b[b["collision"] == 1]
    #b = b[b["collision"] == 0]
    #print(len(a), len(b), len(c))
    #print(compute_trajectory_stats(str(path), p))

    # file = h5py.File(path, "r")
    # group = file["transitions"]
    #
    # train, val, _ = forward_sweep_dataset(path, Path(f"dataset/asds/babbling_master_index.csv"))
    #
    # for idx in train.get_indices():
    #     if not np.any(group["desired_delta_q"][idx]):
    #         print(idx)
    #
    # for idx in val.get_indices():
    #     if not np.any(group["desired_delta_q"][idx]):
    #         print(idx)

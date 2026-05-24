from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from remake.models.datasets import ForwardModelDataset, PretrainTrajectoryModelDataset, InverseModelDataset


def internal_sweep_dataset(
        h5_file_path: str,
        master_index_df_path: str,
        mode_name: str,
        n_transitions: int = 100000,
        target_collision_ratio: float = 0.2,
        target_transition_without_obstacles_ratio: float = 0.1,
        seed: int = 1,
) -> Tuple[Dataset, Dataset, Dataset | None]:
    """
    Creates transition dataset based on given ratios for both the forward and inverse model.

    :param h5_file_path: Path to h5 file with transition data.
    :param master_index_df_path: Master index file path.
    :param mode_name: Name of which model.
    :param n_transitions: Number of transitions.
    :param target_collision_ratio: Ratio of colliding transitions in dataset.
    :param target_transition_without_obstacles_ratio: Number of transitions in environment with no obstacle in dataset.
    :param seed: Seed for random number generator for deterministic behaviour.
    :return: Training and validation transition dataset for sweeps.
    """
    rng = np.random.default_rng(seed)

    master_index_df = pd.read_csv(master_index_df_path)

    obstacle_df, no_obstacle_df = split_by_obstacle(master_index_df)

    obstacle_train_df, obstacle_val_df, _ = split_by_env(rng, obstacle_df)

    obstacle_train_collision_df, obstacle_train_avoidance_df = split_by_collision(obstacle_train_df)
    obstacle_val_collision_df, obstacle_val_avoidance_df = split_by_collision(obstacle_val_df)

    obstacle_train_collision_sample_distribution = distribute_samples(
        obstacle_train_collision_df['setup_id'].nunique(),
        int(n_transitions * target_collision_ratio * (1 - target_transition_without_obstacles_ratio) * 0.9),
        rng,
        (2, 5)
    )

    obstacle_train_avoidance_sample_distribution = distribute_samples(
        obstacle_train_avoidance_df['setup_id'].nunique(),
        int(n_transitions * (1 - target_collision_ratio) * (1 - target_transition_without_obstacles_ratio) * 0.9),
        rng,
        (6, 20)
    )

    obstacle_val_collision_sample_distribution = distribute_samples(
        obstacle_val_collision_df['setup_id'].nunique(),
        int(n_transitions * target_collision_ratio * (1 - target_transition_without_obstacles_ratio) * 0.1),
        rng,
        (1, 5)
    )

    obstacle_val_avoidance_sample_distribution = distribute_samples(
        obstacle_val_avoidance_df['setup_id'].nunique(),
        int(n_transitions * (1 - target_collision_ratio) * (1 - target_transition_without_obstacles_ratio) * 0.1),
        rng,
        (2, 10)
    )

    train_sample_indices = []
    val_sample_indices = []

    train_sample_indices += sample_indices(obstacle_train_collision_df, obstacle_train_collision_sample_distribution,
                                           rng)
    train_sample_indices += sample_indices(obstacle_train_avoidance_df, obstacle_train_avoidance_sample_distribution,
                                           rng)
    val_sample_indices += sample_indices(obstacle_val_collision_df, obstacle_val_collision_sample_distribution, rng)
    val_sample_indices += sample_indices(obstacle_val_avoidance_df, obstacle_val_avoidance_sample_distribution, rng)

    train_no_obstacle_df, val_no_obstacle_df, _ = split_by_env(rng, no_obstacle_df)

    train_no_obstacle_distribution = distribute_samples(
        train_no_obstacle_df['setup_id'].nunique(),
        int(n_transitions * target_transition_without_obstacles_ratio * 0.9),
        rng,
        (10, 30)
    )

    val_no_obstacle_distribution = distribute_samples(
        val_no_obstacle_df['setup_id'].nunique(),
        int(n_transitions * target_transition_without_obstacles_ratio * 0.1),
        rng,
        (1, 10)
    )

    train_sample_indices += sample_indices(train_no_obstacle_df, train_no_obstacle_distribution, rng)
    val_sample_indices += sample_indices(val_no_obstacle_df, val_no_obstacle_distribution, rng)

    if mode_name == "forward":
            return ForwardModelDataset(h5_file_path, train_sample_indices), ForwardModelDataset(h5_file_path,
                                                                                        val_sample_indices), None
    else:
        return InverseModelDataset(h5_file_path, train_sample_indices), InverseModelDataset(h5_file_path,
                                                                                            val_sample_indices), None

def internal_k_fold_cross_validation_dataset_indices(
        master_index_df_path: str,
        n_transitions: int = 525000,
        target_collision_ratio: float = 0.2,
        target_transition_without_obstacles_ratio: float = 0.1,
        k: int = 5,
        seed: int = 1,
) -> List[List[int]]:
    """
    Creates splits of indices for transition datasets during the inverse/forward model cross-validation. Splits are
    indices pointing to the transition data in transition data file.

    :param master_index_df_path: Path to the master index file.
    :param n_transitions: Number of total transitions
    :param target_collision_ratio: Ratio of colliding transitions in dataset.
    :param target_transition_without_obstacles_ratio: Number of transitions in environment with no obstacle in dataset.
    :param k: Number of folds.
    :param seed: Seed for random number generator for deterministic behaviour.
    :return: Folds - List of splits represented by list of indices.
    """

    master_index_df = pd.read_csv(master_index_df_path)
    rng = np.random.default_rng(seed)

    obstacle_df, no_obstacle_df = split_by_obstacle(master_index_df)
    obstacle_collision_df, obstacle_avoidance_df = split_by_collision(obstacle_df)

    k_fold_indices: List[List[int]] = [list() for _ in range(k)]

    obstacle_collision_split_df = split_by_env_into_k_folds(
        obstacle_collision_df, rng, k=k)
    obstacle_avoidance_split_df = split_by_env_into_k_folds(obstacle_avoidance_df, rng, k=k)
    no_obstacle_split_df = split_by_env_into_k_folds(no_obstacle_df, rng, k=k)

    for i in range(k):
        idx = 0
        for split in [obstacle_collision_split_df, obstacle_avoidance_split_df, no_obstacle_split_df]:
            if idx == 0:
                n = n_transitions / k * target_collision_ratio * (1 - target_transition_without_obstacles_ratio)
                r = (0, 1)
            elif idx == 1:
                n = n_transitions / k * (1 - target_collision_ratio) * (1 - target_transition_without_obstacles_ratio)
                r = (0,1)
            else:
                n = n_transitions / k * target_transition_without_obstacles_ratio
                r = (0, 1)

            n = int(n)
            sample_distribution = distribute_samples(
                split[i]['setup_id'].nunique(),
                n,
                rng,
                r
            )

            k_fold_indices[i] += sample_indices(split[i], sample_distribution, rng)
            idx += 1

    return k_fold_indices

def internal_test_sets(
        master_index_df_path: str,
        h5_path: str,
        mode_name: str,
        n_transitions: int = 10000,
) -> Tuple[ForwardModelDataset | InverseModelDataset, ForwardModelDataset | InverseModelDataset, ForwardModelDataset | InverseModelDataset]:
    """
    Creates test datasets for forward/inverse models of transitions split into three categories: transitions in environment
    with no obstacle, transitions in environment with obstacle that lead to collision, and transitions in environment
    with obstacle that are collision-free.

    :param master_index_df_path: Path to the master index file.
    :param h5_path: Path to h5 file with transition data.
    :param mode_name: Name of the model.
    :param n_transitions: Number of transitions.
    :return: Three test datasets for either of the internal models - no obstacle, collision and collision-free trajectories.
    """
    master_index_df = pd.read_csv(master_index_df_path)
    obstacle_df, no_obstacle_df = split_by_obstacle(master_index_df)
    obstacle_collision_df, obstacle_avoidance_df = split_by_collision(obstacle_df)

    no_obstacle_indices = no_obstacle_df['ep_start'].tolist()[:n_transitions]
    obstacle_collision_indices = obstacle_collision_df['ep_start'].tolist()[:n_transitions]
    obstacle_avoidance_indices = obstacle_avoidance_df['ep_start'].tolist()[:n_transitions]

    if mode_name == "forward":
        return (ForwardModelDataset(h5_path=h5_path, datapoint_indices=no_obstacle_indices),
                ForwardModelDataset(h5_path=h5_path, datapoint_indices=obstacle_collision_indices),
                ForwardModelDataset(h5_path=h5_path, datapoint_indices=obstacle_avoidance_indices)
                )
    else:
        return (InverseModelDataset(h5_path=h5_path, datapoint_indices=no_obstacle_indices),
                InverseModelDataset(h5_path=h5_path, datapoint_indices=obstacle_collision_indices),
                InverseModelDataset(h5_path=h5_path, datapoint_indices=obstacle_avoidance_indices)
                )

def trajectory_sweep_dataset(
        h5_file_path: str,
        master_index_df_path: str,
        n_trajectories: int = 13000,
        target_no_obstacle_ratio: float = 0.1,
        seed: int = 1,
) -> Tuple[Dataset, Dataset, Dataset | None]:
    """
    Creates trajectory dataset based on given ratios for trajectory model.

    :param n_trajectories:Number of trajectories.
    :param target_no_obstacle_ratio: Number of trajectories in environment with no obstacle.
    :param h5_file_path: Path to h5 file with trajectory data.
    :param master_index_df_path: Master index file path.
    :param seed: Seed for random number generator for deterministic behaviour.
    :return: Training and validation trajectory dataset for sweeps.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(master_index_df_path)

    # Only good trajectories and appropriate length
    df = df[df["ep_len"] <= 52]
    df = df[df["collision"] == 0]
    df.sort_values(by="episode_id", inplace=True)

    obstacle_df, no_obstacle_df = split_by_obstacle(df)

    no_obstacle_indices = no_obstacle_df["episode_id"].tolist()
    obstacle_indices = obstacle_df["episode_id"].tolist()

    rng.shuffle(no_obstacle_indices)
    rng.shuffle(obstacle_indices)

    no_obstacle_train_range = int(n_trajectories * target_no_obstacle_ratio * 0.9)
    obstacle_train_range = int(n_trajectories * (1 - target_no_obstacle_ratio) * 0.9)
    train_sample_indices = (
        no_obstacle_indices[:no_obstacle_train_range] +
        obstacle_indices[:obstacle_train_range]
    )

    no_obstacle_val_range = int(n_trajectories * target_no_obstacle_ratio * 0.1)
    obstacle_val_range = int(n_trajectories * (1 - target_no_obstacle_ratio) * 0.1)
    val_sample_indices = (
        no_obstacle_indices[no_obstacle_train_range:no_obstacle_train_range + no_obstacle_val_range] +
        obstacle_indices[obstacle_train_range:obstacle_train_range + obstacle_val_range]
    )

    rng.shuffle(train_sample_indices)
    rng.shuffle(val_sample_indices)

    return PretrainTrajectoryModelDataset(h5_file_path, train_sample_indices), PretrainTrajectoryModelDataset(h5_file_path, val_sample_indices), None

def trajectory_k_fold_cross_validation_dataset_indices(
        master_index_df_path: str,
        n_trajectories: int = 12000,
        target_no_obstacle_ratio: float = 0.1,
        k: int = 5,
        seed: int = 1,
) -> List[List[int]]:
    """
    Creates splits of indices for trajectory datasets during the trajectory model cross-validation. Splits are
    indices pointing to the trajectory data in trajectory data file.

    :param n_trajectories: Total number of trajectories.
    :param target_no_obstacle_ratio: Number of transitions in environment with no obstacle in dataset
    :param master_index_df_path: Path to the master index file.
    :param k: Number of folds.
    :param seed: Seed for random number generator for deterministic behaviour.
    :return: Folds - List of splits represented by list of indices.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(master_index_df_path)

    # Only good trajectories and appropriate length
    df = df[df["ep_len"] <= 52]
    df = df[df["collision"] == 0]
    df.sort_values(by="episode_id", inplace=True)

    obstacle_df, no_obstacle_df = split_by_obstacle(df)

    no_obstacle_indices = no_obstacle_df["episode_id"].tolist()
    obstacle_indices = obstacle_df["episode_id"].tolist()

    rng.shuffle(no_obstacle_indices)
    rng.shuffle(obstacle_indices)

    no_obstacle_indices = no_obstacle_indices[:int(n_trajectories * target_no_obstacle_ratio)]
    obstacle_indices = obstacle_indices[:int(n_trajectories * (1 - target_no_obstacle_ratio))]

    k_fold_indices: List[List[int]] = [list() for _ in range(k)]

    for idx, episode_id in enumerate(no_obstacle_indices):
        fold = idx % k
        k_fold_indices[fold].append(episode_id)

    for idx, episode_id in enumerate(obstacle_indices):
        fold = idx % k
        k_fold_indices[fold].append(episode_id)

    for i in range(k):
        rng.shuffle(k_fold_indices[i])

    return k_fold_indices

def build_obstacle_split_datasets(
        h5_path: str,
        master_index_path: str,
        seed: int,
        max_ep_len: int = 52,
) -> Dict[str, PretrainTrajectoryModelDataset]:
    """
    Creates trajectory dataset that is split into no obstacle and no obstacle present environments.

    :param h5_path: Path to h5 file with trajectory data.
    :param master_index_path: Path to master index file.
    :param seed: Seed for random number generator for deterministic behaviour.
    :param max_ep_len: Max length of trajectories.
    :return: Trajectory datasets with no obstacle and with obstacle present environments.
    """
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

def split_by_env(
        rng: np.random.Generator,
        master_index_df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Splits setups into training and validation sets based on given ratios.

    :param rng: Random number generator for deterministic behaviour.
    :param master_index_df: Dataframe with data indices.
    :param train_ratio: Training data ratio.
    :param val_ratio: Validation data ratio.
    :return: Training and validation sets of indices.
    """
    env_indices = master_index_df['setup_id'].unique()
    env_indices = rng.permutation(env_indices)

    n_envs = len(env_indices)
    n_train = int(n_envs * train_ratio)
    n_val = int(n_envs * val_ratio)

    train_env_indices = env_indices[:n_train]
    val_env_indices = env_indices[n_train:n_val + n_train]

    train_df = master_index_df[master_index_df['setup_id'].isin(train_env_indices)].copy()
    val_df = master_index_df[master_index_df['setup_id'].isin(val_env_indices)].copy()

    if val_ratio + train_ratio < 1:
        test_env_indices = env_indices[n_val + n_train:]
        test_df = master_index_df[master_index_df['setup_id'].isin(test_env_indices)].copy()
    else:
        test_df = None

    return train_df, val_df, test_df


def split_by_collision(
        df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits indices based on collision flag.

    :param df: Dataframe with indices.
    :return: Dataframe with collision indices and non-collision indices.
    """
    colliding_episodes = df[df['collision'] == 1].copy()
    collision_free_episodes = df[df['collision'] == 0].copy()

    return colliding_episodes, collision_free_episodes


def distribute_samples(
        n_envs: int,
        n_samples: int,
        rng: np.random.Generator,
        sample_range: Tuple[int, int],
) -> List[int]:
    """
    Distributes samples from each setup.

    :param n_envs: Number of total env setups.
    :param n_samples: Number of total samples to distribute.
    :param rng: Random number generator for deterministic behaviour.
    :param sample_range: Allowed sample range for one setup.
    :return: Distributed samples.
    """
    if n_samples < sample_range[0] * n_envs or n_samples > sample_range[1] * n_envs:
        raise ValueError()

    distribution = [0] * n_envs
    remaining = n_samples - sample_range[0] * n_envs

    for i in range(n_envs):
        slots_left = n_envs - i - 1

        max_take = min(remaining, sample_range[1] - sample_range[0])
        min_take = max(0, remaining - slots_left * (sample_range[1] - sample_range[0]))

        if i == (n_envs - 1):
            take = remaining
        else:
            take = rng.integers(min_take, max_take + 1)

        distribution[i] = take
        remaining -= take

    rng.shuffle(distribution)

    return [x + sample_range[0] for x in distribution]


def sample_indices(
        df: pd.DataFrame,
        distribution: List[int],
        rng: np.random.Generator,
) -> List[int]:
    """
    Samples indices according to distribution.

    :param df: Dataframe with indices.
    :param distribution: Distribution of samples
    :param rng: Random number generator for deterministic behaviour.
    :return: Sampled indices according to distribution.
    """
    indices = []

    for env_id, n_samples in zip(df['setup_id'].unique(), distribution):
        episodes = df[df['setup_id'] == env_id].copy()

        transition_indices = []
        for ep_start, ep_len in zip(episodes['ep_start'], episodes['ep_len']):
            transition_indices += range(ep_start, ep_start + ep_len - 2)
        rng.shuffle(transition_indices)

        indices += transition_indices[:n_samples]

    return indices


def split_by_obstacle(
        df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits indices by obstacle presence.

    :param df: Dataframe with indices.
    :return: Dataframe with no obstacle indices and obstacle indices.
    """
    obstacle_df = df[df['obstacle_active'] == 1].copy()
    no_obstacle_df = df[df['obstacle_active'] == 0].copy()

    return obstacle_df, no_obstacle_df

def split_by_env_into_k_folds(
        df: pd.DataFrame,
        rng: np.random.Generator,
        k: int = 5,
) -> List[pd.DataFrame]:
    """
    Splits indices by environment setup into folds for cross-validation.

    :param df: Dataframe with indices.
    :param rng: Random number generator for deterministic behavior.
    :param k: Number of folds.
    :return: Dataframe with indices for each fold.
    """
    env_indices = df["setup_id"].unique()
    env_indices = rng.permutation(env_indices)

    if len(env_indices) == 0:
        return [df.iloc[0:0].copy() for _ in range(k)]

    if k > len(env_indices):
        raise ValueError(
            f"k={k} is larger than number of unique setup_id values={len(env_indices)}"
        )

    splits = np.array_split(env_indices, k)

    return [
        df[df["setup_id"].isin(split)].copy()
        for split in splits
    ]
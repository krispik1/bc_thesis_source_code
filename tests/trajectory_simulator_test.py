from pathlib import Path
from typing import Dict, Any, List
import time

import commentjson
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wrapper import GymWrapper
from models.trajectory_model.trajectory_model import TrajectoryModel
from models.create_datasets import PretrainTrajectoryModelDataset
from models.trajectory_model.trajectory_model_training import move_batch_to_device


EXECUTION_MODES = [
    #"configuration",
    "end_effector_ik"
]

def reset_env_to_episode_state(
    env,
    setup: Dict[str, np.ndarray],
    goal6D_for_test: np.ndarray,
    obstacle_active: bool,
):

    env.robot.set_magnetization(0)

    if hasattr(env.robot, "release_all_objects"):
        env.robot.release_all_objects()

    env.task.reset_task()

    goal_obj = env.task_objects["goal_state"]
    env.p.resetBasePositionAndOrientation(
        goal_obj.get_uid(),
        goal6D_for_test[:3],
        goal6D_for_test[3:],
    )

    distractors = env.task_objects.get("distractor", [])

    if obstacle_active:
        for d in distractors:
            env.p.resetBasePositionAndOrientation(
                d.get_uid(),
                setup["obstacle6D"][:3],
                setup["obstacle6D"][3:],
            )

            env.p.resetBaseVelocity(
                d.get_uid(),
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
            )

            env.p.changeDynamics(
                d.get_uid(),
                -1,
                mass=0.0,
                linearDamping=1.0,
                angularDamping=1.0,
            )
    else:
        for d in distractors:
            env.p.resetBasePositionAndOrientation(
                d.get_uid(),
                [10.0, 10.0, 10.0],
                [0.0, 0.0, 0.0, 1.0],
            )

    # Reset robot
    set_robot_configuration(
        env,
        setup["initial_configuration"],
    )

    env.p.performCollisionDetection()
    env.p.stepSimulation()

def load_env(cfg_path: str | Path, gui: int = 0):
    with open(cfg_path, "r") as f:
        cfg = commentjson.load(f)

    env_cfg = cfg["env_config"]
    env_cfg["gui"] = gui

    env = GymWrapper(env_cfg, 0.5)
    env.reset()

    return env


def reset_object_pose(env, goal6D, obstacle6D, obstacle_active: bool):
    goal_obj = env.task_objects["goal_state"]

    env.p.resetBasePositionAndOrientation(
        goal_obj.get_uid(),
        goal6D[:3],
        goal6D[3:],
    )

    distractors = env.task_objects.get("distractor", [])

    if obstacle_active:
        for d in distractors:
            env.p.resetBasePositionAndOrientation(
                d.get_uid(),
                obstacle6D[:3],
                obstacle6D[3:],
            )
            env.p.changeDynamics(
                d.get_uid(),
                -1,
                mass=0.0,
                linearDamping=1.0,
                angularDamping=1.0,
            )
    else:
        for d in distractors:
            env.p.resetBasePositionAndOrientation(
                d.get_uid(),
                [10.0, 10.0, 10.0],
                [0.0, 0.0, 0.0, 1.0],
            )

    env.p.stepSimulation()


def set_robot_configuration(env, q):
    robot_uid = env.robot.get_uid()

    revolute_ids = []

    for j in range(env.p.getNumJoints(robot_uid)):
        info = env.p.getJointInfo(robot_uid, j)

        if info[2] == env.p.JOINT_REVOLUTE:
            revolute_ids.append(j)

    revolute_ids = revolute_ids[:7]

    for joint_id, q_i in zip(revolute_ids, q):
        env.p.resetJointState(
            robot_uid,
            joint_id,
            float(q_i),
        )

    env.p.stepSimulation()


def extract_episode_setup(h5_path: str | Path, episode_id: int):
    with h5py.File(h5_path, "r") as h5:
        episodes = h5["episodes"]
        transitions = h5["transitions"]

        ep_start = int(episodes["ep_start"][episode_id])
        ep_len = int(episodes["ep_len"][episode_id])

        initial_idx = ep_start
        final_idx = ep_start + ep_len - 1

        q0 = np.asarray(
            transitions["joints_angles_t"][initial_idx],
            dtype=float,
        )

        goal6D = np.asarray(
            transitions["goal_obj6D_t"][initial_idx],
            dtype=float,
        )

        obstacle6D = np.asarray(
            transitions["obstacle6D_t"][initial_idx],
            dtype=float,
        )

        obstacle6D = np.asarray([-2, -2, -2, -2, -2, -2, -2], dtype=float)

        final_ee = np.asarray(
            transitions["ee6D_t"][final_idx],
            dtype=float,
        )

    return {
        "initial_configuration": q0,
        "goal6D": goal6D,
        "obstacle6D": obstacle6D,
        "final_ee": final_ee,
    }


def has_obstacle_collision(env):
    robot_uid = env.robot.get_uid()
    distractors = env.task_objects.get("distractor", [])

    for d in distractors:
        contacts = env.p.getContactPoints(
            bodyA=robot_uid,
            bodyB=d.get_uid(),
        )

        if len(contacts) > 0:
            return True

    return False


def current_ee_position(env):
    robot_uid = env.robot.get_uid()
    ee_link_id = 6

    ee_state = env.p.getLinkState(
        robot_uid,
        ee_link_id,
        computeForwardKinematics=True,
    )

    return np.asarray(ee_state[0], dtype=float)


def goal_position(env):
    goal_uid = env.task_objects["goal_state"].get_uid()
    pos, _ = env.p.getBasePositionAndOrientation(goal_uid)

    return np.asarray(pos, dtype=float)


def compute_ik_configuration(env, target_pos):
    q = env.robot.calculate_accurate_IK(
        end_effector_pos=np.asarray(target_pos, dtype=float)
    )

    if q is None:
        return None

    q = np.asarray(q, dtype=float)

    if q.shape[0] < 7:
        return None

    if not np.all(np.isfinite(q)):
        return None

    return q[:7]


def trajectory_to_commands(
    env,
    predicted_trajectory: Dict[str, torch.Tensor],
    mode: str,
    final_state: Dict[str, torch.Tensor],
):
    if mode == "configuration":
        q_seq = (
            predicted_trajectory["configuration"][0]
            .detach()
            .cpu()
            .numpy()
        )

        final_q = (
            final_state["configuration"][0]
            .detach()
            .cpu()
            .numpy()
        )

        q_seq = np.concatenate(
            [
                q_seq,
                final_q[None, :],
            ],
            axis=0,
        )

        pred_ee_seq = (
            predicted_trajectory["end_effector_position"][0]
            .detach()
            .cpu()
            .numpy()
        )

        final_ee_pos = (
            final_state["end_effector"][0, :3]
            .detach()
            .cpu()
            .numpy()
        )

        pred_ee_seq = np.concatenate(
            [
                pred_ee_seq,
                final_ee_pos[None, :],
            ],
            axis=0,
        )

        return [q for q in q_seq], pred_ee_seq, 0
    if mode == "end_effector_ik":
        pred_ee_seq = (
            predicted_trajectory["end_effector_position"][0]
            .detach()
            .cpu()
            .numpy()
        )

        final_ee_pos = (
            final_state["end_effector"][0, :3]
            .detach()
            .cpu()
            .numpy()
        )

        pred_ee_seq = np.concatenate(
            [
                pred_ee_seq,
                final_ee_pos[None, :],
            ],
            axis=0,
        )

        commands = []
        ik_failed_waypoints = 0
        previous_q = current_robot_configuration(env)

        for idx, ee_pos in enumerate(pred_ee_seq):

            q = compute_ik_configuration(env, ee_pos)

            if q is None:
                ik_failed_waypoints += 1
                q = previous_q.copy()
            else:
                previous_q = q.copy()

            commands.append(q)

        return commands, pred_ee_seq, ik_failed_waypoints

    raise ValueError(f"Unknown execution mode: {mode}")


def compute_executed_geometry(executed_positions):
    positions = np.asarray(executed_positions, dtype=float)

    if len(positions) < 2:
        return {
            "executed_path_length": float("nan"),
            "executed_mean_step_size": float("nan"),
            "executed_max_step_size": float("nan"),
            "executed_tail_mean_step_size": float("nan"),
            "executed_tail_fraction_static": float("nan"),
            "executed_estimated_effective_length": float("nan"),
            "executed_mean_angle_deg": float("nan"),
            "executed_min_angle_deg": float("nan"),
        }

    step_sizes = np.linalg.norm(
        positions[1:] - positions[:-1],
        axis=1,
    )

    path_length = float(step_sizes.sum())
    mean_step = float(step_sizes.mean())
    max_step = float(step_sizes.max())

    tail_start = int(len(step_sizes) * 2 / 3)
    tail_steps = step_sizes[tail_start:]

    eps = 1e-3

    tail_mean_step = float(tail_steps.mean())
    tail_fraction_static = float((tail_steps < eps).mean())

    consecutive_required = 5
    effective_length = len(step_sizes)

    for t in range(max(0, len(step_sizes) - consecutive_required)):
        if np.all(step_sizes[t:t + consecutive_required] < eps):
            effective_length = t
            break

    angles = []

    if len(positions) >= 3:
        for i in range(1, len(positions) - 1):
            v1 = positions[i - 1] - positions[i]
            v2 = positions[i + 1] - positions[i]

            denom = np.linalg.norm(v1) * np.linalg.norm(v2)

            if denom < 1e-8:
                continue

            cos_theta = np.dot(v1, v2) / denom
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angle = np.degrees(np.arccos(cos_theta))
            angles.append(angle)

    if len(angles) == 0:
        mean_angle = float("nan")
        min_angle = float("nan")
    else:
        mean_angle = float(np.nanmean(angles))
        min_angle = float(np.min(angles))

    return {
        "executed_path_length": path_length,
        "executed_mean_step_size": mean_step,
        "executed_max_step_size": max_step,
        "executed_tail_mean_step_size": tail_mean_step,
        "executed_tail_fraction_static": tail_fraction_static,
        "executed_estimated_effective_length": float(effective_length),
        "executed_mean_angle_deg": mean_angle,
        "executed_min_angle_deg": min_angle,
    }


def compute_predicted_executed_deviation(
    predicted_positions,
    executed_positions,
):
    predicted = np.asarray(predicted_positions, dtype=float)
    executed = np.asarray(executed_positions, dtype=float)

    n = min(len(predicted), len(executed))

    if n == 0:
        return {
            "pred_exec_mean_deviation": float("nan"),
            "pred_exec_final_deviation": float("nan"),
            "pred_exec_max_deviation": float("nan"),
        }

    diff = np.linalg.norm(
        predicted[:n] - executed[:n],
        axis=1,
    )

    return {
        "pred_exec_mean_deviation": float(diff.mean()),
        "pred_exec_final_deviation": float(diff[-1]),
        "pred_exec_max_deviation": float(diff.max()),
    }



def build_obstacle_split_datasets(
    h5_path: str,
    master_index_path: str,
    max_ep_len: int = 52,
):
    df = pd.read_csv(master_index_path)

    df = df[df["ep_len"] <= max_ep_len]
    df = df[df["collision"] == 0]
    df.sort_values(by="episode_id", inplace=True)

    no_obstacle_df = df[df["obstacle_active"] == 0].copy()
    obstacle_df = df[df["obstacle_active"] == 1].copy()

    return {
        "No obstacle": {
            "dataset": PretrainTrajectoryModelDataset(
                h5_path,
                no_obstacle_df["episode_id"].tolist(),
            ),
            "episode_ids": no_obstacle_df["episode_id"].tolist(),
            "episode_success": no_obstacle_df["success"].tolist(),
            "obstacle_active": False,
        },
        "Obstacle": {
            "dataset": PretrainTrajectoryModelDataset(
                h5_path,
                obstacle_df["episode_id"].tolist(),
            ),
            "episode_ids": obstacle_df["episode_id"].tolist(),
            "episode_success": obstacle_df["success"].tolist(),
            "obstacle_active": True,
        },
    }


def load_trajectory_model(checkpoint_path: str | Path, device: str):
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
        head_hidden_dimension=cfg["hidden_head_dimension"],
        n_timesteps=cfg["n_timesteps"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model

def is_object_magnetized(env) -> bool:
    return len(env.robot.magnetized_objects) > 0

def current_robot_configuration(env):
    q = []

    for idx in env.robot.motor_indices:
        q.append(
            env.p.getJointState(
                env.robot.robot_uid,
                idx,
            )[0]
        )

    return np.asarray(q, dtype=float)

@torch.no_grad()
def evaluate_model_in_simulator(
    model,
    env_cfg_path: str | Path,
    h5_path: str | Path,
    category_name: str,
    category_data: Dict[str, Any],
    mode: str,
    device: str,
    seed: int,
    max_episodes: int | None = None,
    success_threshold: float = 0.05,
    gui: int = 0,
):
    dataset = category_data["dataset"]
    episode_ids = list(category_data["episode_ids"])
    episode_success = list(category_data["episode_success"])
    obstacle_active = category_data["obstacle_active"]

    rng = np.random.default_rng(seed)

    indices = np.arange(len(episode_ids))
    rng.shuffle(indices)

    if max_episodes is not None:
        indices = indices[:max_episodes]

    selected_episode_ids = [episode_ids[i] for i in indices]
    selected_dataset_indices = [dataset.datapoint_indices[i] for i in indices]
    selected_episode_success = [episode_success[i] for i in indices]

    dataset = PretrainTrajectoryModelDataset(
        h5_path=str(h5_path),
        datapoint_indices=selected_dataset_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    successes = []
    collisions = []

    final_distances = []
    min_distances = []
    executed_steps = []
    execution_times = []

    geometry_successful = []
    pred_exec_deviation_successful = []

    magnet_successes = []
    raw_magnet_successes = []
    magnet_activation_rates = []
    gt_successes = []
    waypoint_reach_rates = []
    total_repeat_counts = []
    mean_repeat_counts = []
    ik_failure_rates=[]
    env = load_env(env_cfg_path, gui=gui)

    try:
        for local_idx, batch in enumerate(loader):
            episode_id = int(selected_episode_ids[local_idx])
            gt_success = int(selected_episode_success[local_idx])
            gt_successes.append(float(gt_success))

            setup = extract_episode_setup(
                h5_path,
                episode_id,
            )

            goal6D_for_test = setup["goal6D"].copy()

            if gt_success == 1:
                goal6D_for_test[:3] = setup["goal6D"][:3]
            else:
                goal6D_for_test[:3] = setup["final_ee"][:3]

            reset_env_to_episode_state(
                env=env,
                setup=setup,
                goal6D_for_test=goal6D_for_test,
                obstacle_active=obstacle_active,
            )

            batch = move_batch_to_device(batch, device)
            predicted_trajectory = model(batch["input"])

            final_state = {
                "configuration": batch["final_state_configuration"],
                "end_effector": batch["final_state_end_effector"],
                "magnet": batch["final_state_magnet"],
                "goal_obj6D": batch["final_state_goal_obj6D"],
                "obstacle6D": batch["final_state_obstacle6D"],
            }

            commands, predicted_ee_positions, ik_failed_waypoints = trajectory_to_commands(
                env=env,
                predicted_trajectory=predicted_trajectory,
                mode=mode,
                final_state=final_state,
            )

            if commands is None:
                successes.append(0.0)
                collisions.append(0.0)

                raw_magnet_successes.append(0.0)
                magnet_successes.append(0.0)
                magnet_activation_rates.append(0.0)
                waypoint_reach_rates.append(0.0)

                final_distances.append(float("nan"))
                min_distances.append(float("nan"))
                executed_steps.append(0)
                execution_times.append(0.0)

                total_repeat_counts.append(0.0)
                mean_repeat_counts.append(0.0)

                continue

            collided = False
            distances = []
            executed_positions = []

            start_time = time.perf_counter()

            magnet_ever_on = False
            object_ever_magnetized = False

            waypoints_reached = 0
            waypoints_total = len(commands)

            max_control_steps_per_waypoint = 20

            configuration_threshold = 0.02
            ee_threshold = 0.03

            repeat_counts_this_episode = []

            for step_idx, q in enumerate(commands):

                is_last_step = step_idx == len(commands) - 1

                if is_last_step:
                    env.robot.set_magnetization(1)
                    magnet_ever_on = True
                else:
                    env.robot.set_magnetization(0)

                target_ee = predicted_ee_positions[
                    min(step_idx, len(predicted_ee_positions) - 1)
                ]

                reached_command = False
                repeat_count = 0
                for _ in range(max_control_steps_per_waypoint):
                    repeat_count += 1

                    env.step(q)

                    if mode == "configuration":

                        current_q = current_robot_configuration(env)

                        reached_command = (
                                np.linalg.norm(current_q - q)
                                < configuration_threshold
                        )

                    elif mode == "end_effector_ik":

                        current_ee = current_ee_position(env)

                        reached_command = (
                                np.linalg.norm(current_ee - target_ee)
                                < ee_threshold
                        )

                    else:
                        reached_command = True

                    if reached_command:
                        break

                waypoints_reached += int(reached_command)
                repeat_counts_this_episode.append(repeat_count)

                if is_object_magnetized(env):
                    object_ever_magnetized = True

                ee = current_ee_position(env)
                executed_positions.append(ee)

                d_goal = float(
                    np.linalg.norm(
                        ee - goal_position(env)
                    )
                )

                distances.append(d_goal)

                if obstacle_active and has_obstacle_collision(env):
                    collided = True

            elapsed = time.perf_counter() - start_time

            total_repeat_counts.append(
                float(np.sum(repeat_counts_this_episode))
            )

            mean_repeat_counts.append(
                float(np.nanmean(repeat_counts_this_episode))
            )

            ik_failure_rates.append(
                ik_failed_waypoints / max(len(predicted_ee_positions), 1)
            )

            final_distance = distances[-1]
            min_distance = min(distances)

            success = (
                    final_distance < success_threshold
                    and not collided
            )
            raw_magnet_success = object_ever_magnetized
            raw_magnet_successes.append(
                float(raw_magnet_success)
            )
            magnet_success = (
                    object_ever_magnetized
                    and not collided
            )

            successes.append(float(success))
            collisions.append(float(collided))

            magnet_successes.append(float(magnet_success))
            magnet_activation_rates.append(float(magnet_ever_on))

            final_distances.append(final_distance)
            min_distances.append(min_distance)

            executed_steps.append(len(commands))
            execution_times.append(elapsed)

            waypoint_reach_rates.append(
                float(
                    waypoints_reached
                    / max(waypoints_total, 1)
                )
            )

            if len(executed_positions) > 1:
                geometry_successful.append(
                    compute_executed_geometry(executed_positions)
                )

                pred_exec_deviation_successful.append(
                    compute_predicted_executed_deviation(
                        predicted_ee_positions,
                        executed_positions,
                    )
                )

    finally:
        env.close()

    result = {
        "model_category": category_name,
        "mode": mode,
        "seed": seed,
        "ik_waypoint_failure_rate": float(np.mean(ik_failure_rates)),
        "ground_truth_success_rate": float(np.nanmean(gt_successes)),
        "magnet_success_rate": float(np.nanmean(magnet_successes)),
        "magnet_activation_rate": float(np.nanmean(magnet_activation_rates)),
        "success_rate": float(np.nanmean(successes)),
        "collision_rate": float(np.nanmean(collisions)),
        "final_distance_to_goal_mean_all": float(np.nanmean(final_distances)),
        "final_distance_to_goal_std_all": float(np.nanstd(final_distances)),
        "min_distance_to_goal_mean_all": float(np.nanmean(min_distances)),
        "min_distance_to_goal_std_all": float(np.nanstd(min_distances)),
        "executed_steps_mean_all": float(np.nanmean(executed_steps)),
        "execution_time_mean_s_all": float(np.nanmean(execution_times)),
        "n_episodes": len(successes),
        "n_successful": int(np.sum(successes)),
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "waypoint_reach_rate": float(np.nanmean(waypoint_reach_rates)),
        "raw_magnet_success_rate": float(
            np.nanmean(raw_magnet_successes)
        ),
        "mean_action_repeats": float(np.nanmean(mean_repeat_counts)),
        "total_action_repeats": float(np.nanmean(total_repeat_counts)),
    }

    if len(geometry_successful) > 0:
        for key in geometry_successful[0].keys():
            values = np.asarray(
                [g[key] for g in geometry_successful],
                dtype=float,
            )

            result[f"{key}_mean"] = float(np.nanmean(values))
            result[f"{key}_median"] = float(np.nanmedian(values))
            result[f"{key}_iqr"] = float(
                np.nanpercentile(values, 75)
                - np.nanpercentile(values, 25)
            )

    if len(pred_exec_deviation_successful) > 0:
        for key in pred_exec_deviation_successful[0].keys():
            values = np.asarray(
                [g[key] for g in pred_exec_deviation_successful],
                dtype=float,
            )

            result[f"{key}_mean"] = float(np.nanmean(values))
            result[f"{key}_median"] = float(np.nanmedian(values))
            result[f"{key}_iqr"] = float(
                np.nanpercentile(values, 75)
                - np.nanpercentile(values, 25)
            )

    return result


if __name__ == "__main__":
    SEEDS = [1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    CFG_PATH = "training_data_generation.json"

    DATA_DIR = Path(
        "dataset/test_trajectory"
    )

    H5_PATH = str(DATA_DIR / "worker_1.h5")
    MASTER_INDEX_PATH = str(DATA_DIR / "test_trajectory_master_index.csv")

    OUTPUT_DIR = Path("plots/simulator_eval/repeated4")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_PATHS = {
        "TM1_epoch50": "model_0050.pt",
        #"TM2_epoch50": "model_0050.pt",
        #"TM3_epoch50": "model_0050.pt",
        #"TM4_epoch50": "model_0050.pt",
        #"TM5_epoch50": "model_0050.pt",
    }

    datasets = build_obstacle_split_datasets(
        h5_path=H5_PATH,
        master_index_path=MASTER_INDEX_PATH,
    )

    rows: List[Dict[str, Any]] = []

    for model_name, model_path in MODEL_PATHS.items():
        print(f"Loading {model_name}")

        model = load_trajectory_model(
            model_path,
            device=device,
        )

        for mode in EXECUTION_MODES:
            for category_name, category_data in datasets.items():
                for seed in SEEDS:
                    if not "No" in category_name:
                        continue

                    print(
                        f"Evaluating {model_name} | "
                        f"{mode} | {category_name} | seed={seed}"
                    )

                    stats = evaluate_model_in_simulator(
                        model=model,
                        env_cfg_path=CFG_PATH,
                        h5_path=H5_PATH,
                        category_name=category_name,
                        category_data=category_data,
                        mode=mode,
                        device=device,
                        seed=seed,
                        max_episodes=200,
                        success_threshold=0.05,
                        gui=0,
                    )

                    row = {
                        "model": model_name,
                        "execution_mode": mode,
                        "category": category_name,
                        "seed": seed,
                        **stats,
                    }

                    rows.append(row)

    raw_df = pd.DataFrame(rows)

    raw_df.to_csv(
        OUTPUT_DIR / "simulator_trajectory_eval_raw_seeds.csv",
        index=False,
    )

    group_columns = [
        "model",
        "execution_mode",
        "category",
    ]

    metric_columns = [
        "collision_rate",
        "raw_magnet_success_rate",
        "magnet_success_rate",
        #"success_rate",

        #"ground_truth_success_rate",
        "waypoint_reach_rate",
        "mean_action_repeats",
        #"magnet_activation_rate",
        "final_distance_to_goal_mean_all",
        #"min_distance_to_goal_mean_all",
        "executed_steps_mean_all",
        "execution_time_mean_s_all",

        "executed_path_length_mean",
        "executed_mean_step_size_mean",
        #"executed_max_step_size_successful_mean",
        #"executed_tail_mean_step_size_successful_mean",
        #"executed_tail_fraction_static_successful_mean",
        #"executed_estimated_effective_length_successful_mean",
        "executed_mean_angle_deg_mean",
        #"executed_min_angle_deg_successful_mean",

        "pred_exec_mean_deviation_mean",
        #"pred_exec_final_deviation_successful_mean",
        #"pred_exec_max_deviation_successful_mean",
    ]

    summary_rows = []

    for keys, group in raw_df.groupby(group_columns):
        row = {
            "model": keys[0],
            "execution_mode": keys[1],
            "category": keys[2],
        }

        for metric in metric_columns:

            if metric not in group.columns:
                continue

            values = group[metric].to_numpy(dtype=float)

            values = values[np.isfinite(values)]

            if len(values) == 0:
                row[metric] = "nan"
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                continue

            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))

            median = float(np.median(values))

            q25 = float(np.percentile(values, 25))
            q75 = float(np.percentile(values, 75))

            iqr = q75 - q25

            p95 = float(np.percentile(values, 95))

            trimmed_values = np.sort(values)

            k = int(0.05 * len(trimmed_values))

            if len(trimmed_values) > 2 * k:
                trimmed_values = trimmed_values[k:-k]

            trimmed_mean = float(np.nanmean(trimmed_values))

            row[metric] = f"{mean:.6f} ± {std:.6f}"

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

            # row[f"{metric}_median"] = median
            # row[f"{metric}_iqr"] = iqr
            # row[f"{metric}_p95"] = p95
            #
            # row[f"{metric}_trimmed_mean"] = trimmed_mean

        row["n_episodes_per_seed"] = int(group["n_episodes"].max())
        row["n_episodes_total"] = int(group["n_episodes"].sum())
        row["n_seeds"] = len(group)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "simulator_trajectory_eval_summary.csv",
        index=False,
    )

    summary_df.drop(
        columns=[
            c for c in summary_df.columns
            if c.endswith("_mean") or c.endswith("_std")
        ],
        errors="ignore",
    ).to_latex(
        OUTPUT_DIR / "simulator_trajectory_eval_summary.tex",
        index=False,
        escape=False,
    )

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from babbling_explorer.explorer import Explorer
from config import TrainingDatasetGenerationConfig
from dataset_types import Episode, Transition, PlannerMode
from dataset_writer.dataset_schemas import SCHEMA_FROM_STR
from dataset_writer.writer_manager import WriterManager
from trajectory_generator.dataset_generator import DatasetGenerator
from wrapper import GymWrapper


def env_dict(
        env : GymWrapper,
        env_cfg : Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates a dictionary based on Environment Table Schema using information about the environment.

    :param env: Environment.
    :param env_cfg: Environment Configuration.
    :return: Dictionary used to store data into Env Schema Table.
    """

    if env.get_distractors_positions():
        obstacle6D = np.concatenate((env.get_distractors_positions()[0], env.get_distractors_orientations()[0]), dtype=np.float64)
    else:
        obstacle6D = np.asarray([-2, -2, -2, -2, -2, -2, -2], dtype=np.float64)

    return {
        "robot_name": 0,
        "robot_action": 0,
        "robot_init": np.asarray(env_cfg.get("robot_init"), dtype=np.float64),
        "goal_obj_name": 0,
        "goal_obj6D": np.concatenate((env.get_goal_position(), env.get_goal_orientation()), dtype=np.float64),
        "obstacle_name": 0,
        "obstacle6D": obstacle6D,
    }

def episode_dict(
        worker_id: int,
        transition_index: int,
        episode: Episode,
) -> Dict[str, Any]:
    """
    Creates a dictionary based on Episode Table schema using information from Episode.

    :param worker_id: Worker's id that represents the environment's id.
    :param transition_index: Number of transitions stored in the HDF5 file.
    :param episode: Run episode description.
    :return: Dictionary used to store data into Episode Schema Table.
    """
    return {
        "env_id": worker_id,
        "ep_start": transition_index,
        "ep_len": len(episode.transitions) + 1,
        "planner": episode.planner_policy,
        "mode": episode.planner_mode,
        "collision": episode.episode_collision,
        "success": episode.success
    }

def transition_dict(
        transition: Transition
) -> Dict[str, Any]:
    """
    Creates a dictionary based on Transitions Table schema using information from Transition and its vectors.

    :param transition: Transition description.
    :return: Dictionary used to store data into Transitions Schema Table.
    """
    if transition.state_t.obstacle6D.size != 0:
        obstacle6D_t = transition.state_t.obstacle6D
        obstacle6D_t1 = transition.state_t1.obstacle6D
    else:
        obstacle6D_t = np.asarray([-2, -2, -2, -2, -2, -2, -2], dtype=np.float64)
        obstacle6D_t1 = np.asarray([-2, -2, -2, -2, -2, -2, -2], dtype=np.float64)

    return {
        "joints_angles_t": transition.state_t.joints_angles,
        "ee6D_t": transition.state_t.end_effector6D,
        "goal_obj6D_t": transition.state_t.goal_object6D,
        "obstacle6D_t": obstacle6D_t,
        "mgt_t": transition.state_t.magnet_state,
        "joints_angles_t1": transition.state_t1.joints_angles,
        "ee6D_t1": transition.state_t1.end_effector6D,
        "goal_obj6D_t1": transition.state_t1.goal_object6D,
        "obstacle6D_t1": obstacle6D_t1,
        "mgt_t1": transition.state_t1.magnet_state,
        "desired_delta_q": transition.action.desired_delta_q,
        "delta_q": transition.action.delta_q,
        "delta_mgt": transition.action.delta_mgt,
        "collision": transition.step_collision
    }

def final_observation_transition_dict(
        transition: Transition
) -> Dict[str, Any]:
    """
    Creates a dictionary based on Transitions Table schema using information from Transition and its vectors, and
    substituting action with zero vector representation.

    :param transition: Last transition description.
    :return: Dictionary used to store last state data from episode into Transitions Schema Table.
    """
    if transition.state_t1.obstacle6D.size != 0:
        obstacle6D = transition.state_t1.obstacle6D
    else:
        obstacle6D = np.asarray([-2, -2, -2, -2, -2, -2, -2], dtype=np.float64)

    return {
        "joints_angles_t": transition.state_t1.joints_angles,
        "ee6D_t": transition.state_t1.end_effector6D,
        "goal_obj6D_t": transition.state_t1.goal_object6D,
        "obstacle6D_t": obstacle6D,
        "mgt_t": transition.state_t1.magnet_state,
        "joints_angles_t1": transition.state_t1.joints_angles,
        "ee6D_t1": transition.state_t1.end_effector6D,
        "goal_obj6D_t1": transition.state_t1.goal_object6D,
        "obstacle6D_t1": obstacle6D,
        "mgt_t1": transition.state_t1.magnet_state,
        "desired_delta_q": np.zeros(7),
        "delta_q": np.zeros(7),
        "delta_mgt": 0,
        "collision": 0
    }

def setup_worker_logger(
        worker_id: int,
        mode: str
) -> logging.Logger:
    """
    Sets up logger for the worker using library logging.

    :param worker_id: Worker's id.
    :return: Logger.
    """
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"logs/{mode}/worker_{worker_id}.log")
    # Logs time, id of the worker and given message
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [worker %(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def worker(
        worker_id: int,
        config: TrainingDatasetGenerationConfig,
        mode: str,
        n_trajectories_per_worker: int = 0,
        n_babbles_per_worker: int = 0
) -> None:
    """
    Function that represents one worker. A worker creates its writers, environment and logger and closes them after
    finishing. It runs data generation episodes for both babbling observations with Explorer, and trajectories with
    DatasetGenerator.

    After each episode, worker logs trajectory's index or number of observations based on the current mode.

    :param n_babbles_per_worker: Number of babbling observations per worker.
    :param n_trajectories_per_worker: Number of trajectories per worker.
    :param mode: Babbling or trajectory mode.
    :param worker_id: Worker's id.
    :param config: Config to initialize the dataset generation.
    """

    writers : Dict[str, WriterManager] = {}
    logger = setup_worker_logger(worker_id, mode)
    env = GymWrapper(config.env_config, config.magnet_probability)
    cleaned_up = False

    # Methods to handle shutdown or kill of the process for consistency in dataset
    def cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True

        for name, w in list(writers.items()):
            try:
                w.close()
            except Exception as e:
                print(f"Worker {worker_id}: failed to close writer '{name}': {e}", file=sys.stderr)

        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Worker {worker_id}: failed to close env: {e}", file=sys.stderr)

    def handle_shutdown(signum, frame) -> None:
        signame = signal.Signals(signum).name
        if logger is not None:
            try:
                logger.warning(f"Worker {worker_id} received {signame}, shutting down.")
            except Exception:
                pass

        cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Initialize writers, environment and logger
    try:
        for schema in [SCHEMA_FROM_STR[s] for s in config.schemas]:
            writers[schema.name] = WriterManager(
                schema=schema,
                path=Path(f"{config.path}/{mode}/worker_{worker_id}.h5"),
                chunks=config.chunks,
                buffer_size=config.buffer_size
            )
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        raise

    try:
        # Count total transitions stored in HDF5 file
        total_transitions = 0 + writers["transitions"].get_size()

        # Log start in which mode and count trajectories/observations based on mode
        logger.info(f"Worker {worker_id} starts generating in {mode} mode.")
        counter = 0

        if mode == "babbling":
            n_runs = n_babbles_per_worker
            n_episodes = 1
        else:
            n_runs = n_trajectories_per_worker
            n_episodes = 1

        avoid_collide_ratio = [0, 0]

        if mode == "babbling":
            env.reset_until_reachable()
            # If babbling, use explorer
            generator = Explorer(
                env=env,
                cfg=config,
            )
        else:
            # If trajectory, use generator
            generator = DatasetGenerator(
                env=env,
                cfg=config
            )

        # Generate until we have enough data
        while counter < n_runs:

            n = 0

            while n < n_episodes:
                if mode == "babbling":
                    if n == 0:
                        planner_mode = "next"
                    else:
                        planner_mode = "prev"
                else:
                    planner_mode = PlannerMode.COLLIDE

                time.sleep(0.9)
                episode = generator.collect_data(planner_mode)
                time.sleep(0.9)
                # If no episode was run (no trajectory was found, babbling episode did not meet conditions, etc.), skip
                if episode is None:
                    continue

                env = generator.env

                # Store environment properties
                writers["env"].save_data(env_dict(env, config.env_config))

                # Store episode data, using total_transitions as pointer to start of episode in transitions table
                writers["episodes"].save_data(episode_dict(worker_id, total_transitions, episode))
                # + 1 for last state + zero action observation
                total_transitions += len(episode.transitions) + 1

                # Store episode state t + action t observations and last state + zero action observation
                for transition in episode.transitions:
                    writers["transitions"].save_data(transition_dict(transition))
                writers["transitions"].save_data(final_observation_transition_dict(episode.transitions[len(episode.transitions) - 1]))

                # Update counter based on mode and log info
                if mode == "babbling":
                    counter += len(episode.transitions)
                    n += len(episode.transitions)
                    logger.info(f"Worker {worker_id} generated {counter} babbling transitions.")
                else:
                    n += 1
                    counter += 1
                    if episode.episode_collision:
                        avoid_collide_ratio[1] += 1
                    else:
                        avoid_collide_ratio[0] += 1
                    logger.info(f"Worker {worker_id} generated trajectory episode {counter}.")
    except SystemExit:
        raise
    finally:
        # Close writers, environment and log finish
        for writer in writers.values():
            writer.close()
        env.close()
        print(f"Worker {worker_id} finished.")
        logger.info(f"Worker {worker_id} finished.")

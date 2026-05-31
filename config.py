from pydantic import BaseModel
from typing import List, Any, Dict, Tuple


class TrainingDatasetGenerationConfig(BaseModel):
    """
    Configuration for generating training dataset.

    Environment configuration:
        (GymEnv)
        - env_config - json dictionary from MyGym,
        (Gym Wrapper)
        - magnet_probability - probability of randomly turning on the magnet for environment step,
        (General)
        - obstacle_present - obstacle absence flag

    Mode of data generation configuration:
        (ParentLauncher)
        - modes - modes of generation (whether trajectories or babbling is generated),
        - n_datapoints_per_mode - number of datapoints (trajectories or babbling observations) per mode generated,
        - n_episodes_per_setup_per_mode - number of episodes per one setup of environment (poses of the objects in env),
        for babbling, horizon of one episode is randomly chosen from interval_n_steps_per_exploration, for trajectories,
        horizon of one episode is length of the generated trajectory.

    Trajectory dataset generator configuration:
        (Planner)
        - target_collision_ratio - ratio of colliding vs non-colliding trajectories to be reached in generated dataset,
        - target_n_waypoints_ratio - ratio of waypoint numbers used to plan trajectories to be reached in generated dataset,
        (Episode Runner)
        - episode_horizon - maximum number of steps for episodes
        - max_steps_per_transitions - maximum number of steps per transition between two configuration during episode run,
        (RRT policy)
        - max_step_size - maximum length of step in a transition,
        - checked_edge_substeps - number of substeps checked to validate an edge in RRT planning,
        - goal_bias - probability of choosing the goal configuration as target configuration in RRT tree,
        - max_rrt_iterations - maximum number of RRT iterations allowed in RRT planning,
        (Smoothing)
        - max_shortcut_length - maximum length of shortcut between two configurations when shortening path in RRT planning,
        - max_degree_angle_turn - maximum degree angle turn of end effector in RRT planning,
        - min_n_nodes_in_path - minimum number of nodes in RRT planned path,
        - max_passes_during_smoothing - maximum number of passes during smoothing of RRT planned path,
        - max_distance_during_smoothing_displacement - maximum distance a configuration can be moved during smoothing
        of RRT planned path,
        - lam - aggressiveness of smoothing of RRT planned path,
        - max_restarts_rrt_planning - maximum tries to use RRT to plan path for given configurations.

    Explorer configuration:
        (Explorer)
        - n_observations - number of observations current state + action -> next state to generate per env,
        - interval_n_steps_per_exploration - interval from which we choose number of steps per exploration episode,
        - region_ratio - ratio of regions (far, near, contact) explored to be reached in dataset,
        - n_candidates - number of candidate actions from which we pick next action based on uncertainty via disagreement,
        (Proxy Ensemble)
        - n_models - number of models in proxy ensemble,
        - hidden - number of hidden layers in models,
        - lr - learning rate for optimizer,
        - batch_size - batch size for training,
        - train_steps_per_env_step - training step for models per number of environment steps,
        (Replay Buffer)
        - replay_capacity - capacity of replay buffer from which we batch sample data for training of the proxy ensemble,
        (Poisson Sampler)
        - max_poisson_sample_tries - maximum tries to sample initial position by Poisson Disk approach,
        - d_near - Far-Near region boundary,
        - d_contact - Near-Contact region boundary,
        - q_min - minimum value allowed to be sampled for each joint angle,
        - q_max - maximum value allowed to be sampled for each joint angle,
        - r - clearance distance used in Poisson Disk sampling.

    HDF5 Writer configuration:
        (Writer Manger)
        - n_writers - number of writers
        - schemas - list of schemas for each writer,
        (HDF5 Writer)
        - path - os path for HDF5 file utilized by the writer,
        - chunks - size of chunks in HDF5 files,
        (Writer Buffer)
        - buffer_size - size of buffer used by HDF5 writers.

    Multiprocessing config:
        - n_workers_per_mode - number of processes per data generation mode.
    """
    # Environment configuration
    env_config: Dict[str, Any]

    magnet_probability: float
    obstacle_present: int

    # Different mode/dataset generation configurations
    modes: List[str]
    n_datapoints_per_mode: List[int]
    n_episodes_per_setup_per_mode: List[int]

    # Trajectory generator configuration
    target_collision_ratio: float
    target_n_waypoints_ratio: List[float]

    episode_horizon: int
    max_steps_per_transition: int

    goal_bias: float
    max_rrt_iterations: int

    max_shortcut_length: float
    max_degree_angle_turn: float
    min_n_nodes_in_path: int
    max_passes_during_smoothing: int
    max_distance_during_smoothing_displacement: float
    lam: float
    max_restarts_rrt_planning: int

    # Explorer configuration
    interval_n_steps_per_exploration: Tuple[int, int]
    region_ratio: Tuple[float, float, float]
    n_candidates: int

    n_models: int
    hidden: int
    lr: float
    batch_size: int
    train_steps_per_env_step: int

    replay_capacity: int

    max_poisson_sample_tries: int
    d_near: float
    d_contact: float
    q_min: List[float]
    q_max: List[float]
    r: float

    # HDF5 writers configuration
    n_writers: int
    schemas: List[str]

    path: str
    chunks: int

    buffer_size: int

    # Multiprocessing configuration
    n_workers_per_mode: List[int]
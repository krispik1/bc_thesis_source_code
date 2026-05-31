from typing import Optional

from config import TrainingDatasetGenerationConfig
from dataset_types import Episode, PlannerMode
from trajectory_generator.episode_runner import EpisodeRunner
from trajectory_generator.planner import Planner
from wrapper import GymWrapper

class DatasetGenerator:

    def __init__(
            self,
            env: GymWrapper,
            cfg: TrainingDatasetGenerationConfig
    ):
        """
        Dataset generator that utilizes various planners to generate trajectories which are followed in episodes from
        which we get observations and actions.

        :param env: Gym environment with added functionalities.
        :param cfg: Config file.
        """
        self.env = env
        self.cfg = cfg

        # Obstacle flag
        self.no_obstacle = not bool(cfg.obstacle_present)

        # Initialize planner and episode generator
        self.planner = Planner(env, self.cfg)
        self.episode_gen = EpisodeRunner(env, self.cfg)

        # Bookkeeping for balancing the ratio
        self._desired_collision_ratio = self.cfg.target_collision_ratio
        self._n_collision = 0
        self._n_non_collision = 0
        self.episode_index = 0
        self.n_waypoints_in_trajectories = [0, 0, 0, 0]

    def update_env(
            self
    ) -> None:
        """
        Updates env for planner and episode runner.
        """
        self.planner.env = self.env
        self.episode_gen.env = self.env

    def collect_data(
            self,
            change_setup: bool,
    ) -> Optional[Episode]:
        """
        Runs data collection episodes, each with new trajectory the robot follows. These episodes are divided into
        collision and avoidance episodes, the former containing episodes where trajectory led to a collision with
        a distractor/occlusion, and the latter containing successful trajectories. We try to balance the ratio by
        choosing modes in which the episodes are run.

        Based on the number of episodes given for each planner, the dataset uses multiple planners:
            - RRT - Trajectory created by Rapidly-exploring Random Trees,
            - Waypoint + RRT - generated waypoints are used for detouring, increases randomization.

        :param change_setup: Update poses of objects in the environment to change setup.
        :return: Collected data of episodes represented by list of observations given by transitions (s_t -> a_t -> s_t1)
        and collision flag (True only if collision), information about the planner and success of the robot if trajectory
        was found.
        """

        # If episode needs new setup, update env
        if change_setup:
            self.env.reset_until_reachable()
            self.update_env()
        else:
            self.env.soft_reset_robot_only()

        current_collision_ratio = 0
        # Prevent division by 0
        if self.episode_index != 0:
            current_collision_ratio = self._n_collision / self.episode_index

        if current_collision_ratio < self._desired_collision_ratio and not self.no_obstacle:
            mode = PlannerMode.COLLIDE
        else:
            mode = PlannerMode.AVOID

        trajectory, n_waypoints = self.planner.plan(
            current_collision_ratio=current_collision_ratio,
            current_n_waypoints_ratio = self.n_waypoints_in_trajectories,
            planner_mode=mode
        )

        # If no trajectory was found, return nothing
        if trajectory is None:
            return None

        episode = self.episode_gen.run_planned_episode(
            trajectory=trajectory,
            planner_mode=self.planner.planner_mode
        )

        # If episode is valid, count it
        if len(episode.transitions) > 0 and mode == episode.episode_collision:

            # Bookkeep for choosing modes and number of waypoints
            self.n_waypoints_in_trajectories[n_waypoints] += 1
            if episode.episode_collision:
                self._n_collision += 1
            else:
                self._n_non_collision += 1
            self.episode_index += 1

            return episode

        return None
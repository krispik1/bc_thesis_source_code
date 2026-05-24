import random
from typing import List, Tuple, Optional

import numpy as np

from remake.config import TrainingDatasetGenerationConfig
from remake.dataset_types import PlannerMode
from remake.trajectory_generator.policies.rrt_policy import RRTTrajectoryGenerator
from remake.wrapper import GymWrapper

class Planner:
    def __init__(
            self,
            env: GymWrapper,
            cfg: TrainingDatasetGenerationConfig,
    ) -> None:
        """
        Planner that utilizes various policies to generate trajectories which are followed in episodes from
        which we get observations and actions.

        :param env: Gym environment with added functionalities.
        :param cfg: Config file.
        """
        self.env = env
        self.cfg = cfg

        self.planner_mode = PlannerMode.DIRECT

        # Initialize trajectory generator
        self.rtt_gen = RRTTrajectoryGenerator(env, self.cfg)

        self.target_collision_ratio = self.cfg.target_collision_ratio
        self.target_n_waypoints_ratio = self.cfg.target_n_waypoints_ratio

    def _choose_mode(
            self,
            current_collision_ratio: float
    ) -> None:
        """
        Used to choose a mode of the planner, whether it plans a colliding, avoiding and direct trajectory.
        """
        # Always try direct approach first
        if current_collision_ratio == 0:
            self.planner_mode = PlannerMode.DIRECT

        # Based on target vs current ratio, bias the modes but allow both
        if current_collision_ratio < self.target_collision_ratio:
            self.planner_mode = random.choices(
                [PlannerMode.COLLIDE, PlannerMode.AVOID],
                weights=[0.7, 0.3],
                k=1
            )[0]
        else:
            self.planner_mode = random.choices(
                [PlannerMode.AVOID, PlannerMode.COLLIDE],
                weights=[0.7, 0.3],
                k=1
            )[0]

    def _choose_n_waypoints(
            self,
            current_n_waypoints_ratio: List[int],
    ) -> int:
        """
        Chooses number of waypoints for generated trajectories based on ratio in dataset.

        :param current_n_waypoints_ratio: Current ratios.
        :return: Number of waypoints.
        """
        total_trajectories = sum(current_n_waypoints_ratio)

        current_percentual_ratios = np.array(current_n_waypoints_ratio) / total_trajectories
        deficit = self.target_n_waypoints_ratio - current_percentual_ratios

        return int(np.argmax(deficit))

    def plan(
            self,
            current_n_waypoints_ratio: List[int] = None,
            current_collision_ratio: float = 0.0,
            planner_mode: PlannerMode = None
    ) -> Tuple[Optional[List[np.ndarray]], int]:
        """
        Generates trajectory based on chosen planner algorithm and its mode given in joint angles:
            - direct rrt - [q_start, q_1, ..., q_goal] where q_goal = q_goal_obj_pos if "avoid", q_goal = q_collision if "collide"
            and collision happens before reaching q_goal in configuration q_collision, and q_goal = q_d_colliding if "collide",
            or None if no trajectory was found,
            - rrt + waypoints - combines waypoint calculation from which we get waypoints representing trajectory T and
            through pairwise iteration (q_start, q_goal) from T, we plan trajectory [q_start, q_1, ..., q_goal] and join
            them. These waypoints vary in numbers and can be used in colliding or avoiding modes.

        :param current_n_waypoints_ratio: Ratio of number of waypoints in trajectories.
        :param current_collision_ratio: Current ratio of colliding trajectories to avoiding trajectories in the dataset.
        :param planner_mode: Possibility to choose the mode of the planner for testing.
        :return: Joints angle trajectory if one was found and number of detour waypoints in it.
        """
        start_pos = self.env.robot.calculate_accurate_IK(end_effector_pos=self.env.ee_init_pose)
        goal_poses = self.env.goal_poses

        # Choose AVOID or COLLIDE based on ratio in generated dataset so far
        if planner_mode is None:
            self._choose_mode(current_collision_ratio)
        else:
            self.planner_mode = planner_mode

        # Choose number of waypoints based on ratio in generated dataset so far
        if sum(current_n_waypoints_ratio) == 0:
            n_waypoints = 0
        else:
            n_waypoints = self._choose_n_waypoints(current_n_waypoints_ratio)

        # If COLLIDE, choose a goal pose leading to collision, otherwise choose goal pose to reach goal object
        if self.planner_mode == PlannerMode.COLLIDE:
            goal_poses = [self.env.robot.calculate_accurate_IK(end_effector_pos=self.env.find_ee_position_for_link_collision())]
        else:
            goal_poses = goal_poses

        waypoints = [start_pos]

        # Get detour waypoints
        detour_waypoints = self.env.generate_curved_waypoints_one_obstacle(n_waypoints=n_waypoints)
        self.env.soft_reset_robot_only()
        n_detour_waypoints = len(detour_waypoints)
        for w in detour_waypoints:
            self.env.show_waypoint_marker(w)
        if detour_waypoints:
            waypoints += [self.env.robot.calculate_accurate_IK(end_effector_pos=detour_waypoint) for detour_waypoint in detour_waypoints]

        # For each goal pose, try to find trajectory
        for goal_pose in goal_poses:
            self.env.soft_reset_robot_only()

            # Combine waypoints with goal pose of the robot
            waypoints_for_goal_pose = np.concatenate((waypoints, [goal_pose]))

            traj = []

            self.env.soft_reset_robot_only()

            for i in range(len(waypoints_for_goal_pose) - 1):
                traj += self.rtt_gen.plan(
                    waypoints_for_goal_pose[i],
                    waypoints_for_goal_pose[i + 1],
                    self.planner_mode
                )

            # If trajectory was found, return it with number of waypoints
            if traj:
                return traj, n_detour_waypoints

        return None, n_detour_waypoints
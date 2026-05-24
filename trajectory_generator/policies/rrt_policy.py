from typing import Tuple, List

import numpy as np

from config import TrainingDatasetGenerationConfig
from dataset_types import PlannerMode
from trajectory_generator.policies.one_euro_filter import OneEuroFilter
from wrapper import GymWrapper
from geometry_helper import dist


def _is_jagged_triangle(
        q_0: np.ndarray,
        q_1: np.ndarray,
        q_2: np.ndarray,
        angle_degree: float = 60.0
) -> bool:
    """
    Tests whether the joints configurations create a segment of trajectory that looks jagged.

    :param q_0: First joints configuration represented by angles.
    :param q_1: Second joints configuration represented by angles.
    :param q_2: Third joints configuration represented by angles.
    :param angle_degree: Maximum turn angle in degrees which the robot's end effector is allowed to make for smoother
    looking trajectory.
    :return: True only if the triangle formed by given joints configuration is too sharp.
    """
    v1 = q_1 - q_0
    v2 = q_2 - q_1
    n1 = dist(q_0, q_1)
    n2 = dist(q_1, q_2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0 >= angle_degree

    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(c)))

    return ang >= angle_degree


def _nearest_index(
        nodes: List[np.ndarray],
        q: np.ndarray
) -> int:
    """
    Finds index of the nearest neighbour for given configuration q in RRT represented by nodes.

    :param nodes: RRT given in nodes.
    :param q: Joint configuration.
    :return: Index of the nearest neighbour for given configuration.
    """

    distances = [dist(n, q) for n in nodes]
    return int(np.argmin(distances))


def _trace_path(
        nodes: List[np.ndarray],
        parents: List[int],
        configuration_index: int
) -> List[np.ndarray]:
    """
    Traces the path from the shared configuration to the root configuration, reversing it to give us a path from root
    to shared configuration.

    :param nodes: RRT given in nodes.
    :param parents: RRT nodes that are extended by an edge already.
    :param configuration_index: Index of the shared configuration node in both RRTs or colliding configuration.
    :return: Path from root configuration -> shared configuration
    """
    path = []

    # Traverse the tree until reaching root configuration that has no parent
    while configuration_index != -1:
        path.append(nodes[configuration_index])
        configuration_index = parents[configuration_index]

    # Reverse it to get trajectory root configuration -> shared configuration
    path.reverse()
    return path


def _merge_paths(
        nodes_a: List[np.ndarray],
        parents_a: List[int],
        index_a: int,
        nodes_b: List[np.ndarray],
        parents_b: List[int],
        index_b: int,
) -> List[np.ndarray]:
    """
    Merges paths after shared configuration between both trees has been found. Path is traced and reversed in both
    trees but path rooted in goal configuration is reversed again as root configuration is the goal configuration
    which should be last.

    :param nodes_a: RRT given in nodes rooted in initial configuration.
    :param parents_a: RRT nodes rooted in initial configuration that are extended by an edge already.
    :param index_a: Index of shared configuration in RRT rooted in initial configuration.
    :param nodes_b: RRT given in nodes rooted in goal configuration.
    :param parents_b: RRT nodes rooted in goal configuration that are extended by an edge already.
    :param index_b: Index of goal configuration in RRT rooted in goal configuration.
    :return: Trajectory initial configuration -> goal configuration given in joint angles.
    """

    path_a = _trace_path(
        nodes_a,
        parents_a,
        index_a
    )

    path_b = _trace_path(
        nodes_b,
        parents_b,
        index_b
    )

    # Reverse path from goal configuration again as it was already correctly ordered before trace path reversal
    path_b.reverse()

    # Avoid duplicating the connection point if they are identical
    if dist(path_a[-1], path_b[0]) < 1e-9:
        path_b = path_b[1:]

    # Merge paths
    return path_a + path_b

class RRTTrajectoryGenerator:

    def __init__(
            self,
            env: GymWrapper,
            cfg: TrainingDatasetGenerationConfig
    ):
        """
        Generator used to generate dataset of trajectories in environment with obstacles through Rapidly exploring
        Random Trees.

        :param env: Gym environment with added functionalities.
        :param cfg: Config file.
        """
        self.env = env
        self.cfg = cfg

        self.goal_bias = self.cfg.goal_bias

        # Joint limits provide sample space for configurations and actions dq
        self.lows, self.highs = self.env.robot.joints_limits

        # Action space is given by maximum velocity and time step of environment step - max difference in configurations
        # for one step must be executable in one environment step
        self.dt = self.env.p.getPhysicsEngineParameters()["fixedTimeStep"]
        v_max = np.asarray(self.env.robot.joints_max_velo, dtype=float)
        # First float = safety clearance
        self.dq_max = 0.5 * self.cfg.env_config["action_repeat"] * v_max * self.dt

    def _get_next_q_from_action_executable_in_sim_timestep(
            self,
            q: np.ndarray,
            dq: np.ndarray,
    ) -> np.ndarray:
        """
        Returns configuration after added action that is executable in one environment step.

        :param q: Start configuration to which we want to add action dq.
        :param dq: Action.
        :return: Configuration after adding action that respects time step of environment.
        """

        return q + np.clip(dq, -self.dq_max, self.dq_max)

    def _sample_joints_angles(
            self
    ) -> np.ndarray:
        """
        Used to sample joint configuration from the joint limits of the robot.

        :return: Configuration of joints represented by angles.
        """
        return np.array(
            [np.random.uniform(low, high) for low, high in zip(self.lows, self.highs)],
            dtype=float,
        )

    def _steer(
            self,
            q_from: np.ndarray,
            q_to: np.ndarray
    ) -> np.ndarray:
        """
        Steers edge from start configuration to target configuration. It respects the time step of simulator so the action
        is executable during one step of simulator.

        :param q_from: Start configuration.
        :param q_to: Target configuration.
        :return: New configuration that extends tree.
        """

        return self._get_next_q_from_action_executable_in_sim_timestep(q_from, q_to - q_from)


    def _check_validity(
            self,
            q: np.ndarray,
            allow_hit: bool
    ) -> Tuple[bool, bool]:
        """
        Checks whether the movement performed by the robot is allowed.

        When collision is present, detected by the physics engine, the movement is only valid if impact with obstacle is
        allowed as it is allowed in collision mode of planner.

        :param q: Joint configuration for which we check validity.
        :param allow_hit: True during collision planning which allows impacts.
        :return: Is movement valid and did collision happen.
        """
        self.env.set_robot_configuration_kinematic(q)
        collision_flag = self.env.check_robot_distractor_collision()

        return (allow_hit or not collision_flag), collision_flag

    def _edge_valid(
            self,
            q_from: np.ndarray,
            q_to: np.ndarray,
            mode: PlannerMode
    ) -> Tuple[bool, bool]:
        """
        Checks edge's validity based on conditions given by the planner mode. In collision mode, edge must exist,
        whether it leads to collision or not. In avoid mode, the edge must exist and not be colliding.

        It interpolates vector to check intermediate steps, dividing it into substeps based on size.


        :param q_from: Start configuration.
        :param q_to: Goal configuration.
        :param mode: Planner mode.
        :return: Valid only if edge is valid and collision flag.
        """

        collision_flag = False

        # Get vector representation of edge
        dq = np.abs(q_to - q_from)

        # Steps into which we interpolate
        max_joint_interp = 0.01

        # Number of steps depending on highest joint angle difference of start and goal configuration
        n_steps = max(2, int(np.ceil(np.max(dq) / max_joint_interp)))

        for i in range(n_steps + 1):
            # Parameter to scale to substep
            t = i / float(n_steps)
            q = q_from + t * (q_to - q_from)

            # Check if substep is valid
            valid, collision = self._check_validity(
                q,
                allow_hit=(mode != PlannerMode.AVOID)
            )

            collision_flag = collision or collision_flag
            if not valid:
                return False, collision_flag

        return True, collision_flag

    def _shortcut_valid(
            self,
            q_0: np.ndarray,
            q_1: np.ndarray,
            mode: PlannerMode,
    ) -> bool:
        """
        Tests if new shortcut created during trajectory is valid.

        :param q_0: Starting joints configuration.
        :param q_1: End joints configuration.
        :param mode: Mode of the planner.
        :return: True only if the new edge representing the shortcut is valid.
        """
        if dist(q_0, q_1) > self.cfg.max_shortcut_length:
            return False

        ok, _ = self._edge_valid(q_0, q_1, mode)

        return ok

    def _extend(
            self,
            nodes: List[np.ndarray],
            parents: List[int],
            q_target: np.ndarray,
            mode: PlannerMode,
    ) -> Tuple[str, int, bool]:
        """
        Extends RRT given in nodes towards the target configuration q_target, respecting the conditions of planner mode.

        The RRT is extended by creating an edge from nearest configuration to the target configuration. As this edge
        cannot exceed maximum step size in length, we calculate new configuration that steers the extension towards the
        target configuration.

        Edge extended configuration -> new configuration must be checked for validity, meaning if it is of appropriate
        length and whether it doesn't break conditions depending on the planner's mode. If the planner is in colliding
        mode, the edge can lead to a colliding state, otherwise it must not collide.

        Depending on certain checks, extension can have one of three statuses:
            - trapped - the edge is not valid,
            - reached - the edge extends the tree to target pose or is one step away from reaching it respecting max step size,
            - advanced - the edge extends the tree, but does not reach the target configuration.

        :param nodes: RRT given in nodes.
        :param parents: RRT nodes that are extended by an edge already.
        :param q_target: Target configuration to which we want to extend.
        :param mode: Mode of the planner.
        :return: Status of extension, index of new node in RRT, and collision flag
        """
        # Find nearest neighbour to the target configuration and extend from there
        index_near = _nearest_index(nodes, q_target)
        q_near = nodes[index_near]

        # Get configuration that steers towards target
        q_new = self._steer(q_near, q_target)

        # Check if edge between configuration from which we extend and new configuration is okay depending on planner's mode
        edge_ok, collision_flag = self._edge_valid(q_near, q_new, mode)
        if not edge_ok:
            return "trapped", -1, collision_flag

        # Extend the tree using the new node and add extended node to parents
        nodes.append(q_new)
        parents.append(index_near)
        new_index = len(nodes) - 1

        # Check if the extension reaches the original target by being in identical position
        if dist(q_new, q_target) < 1e-9:
            return "reached", new_index, collision_flag

        # Else the tree grew by node in new_index
        return "advanced", new_index, collision_flag

    def _connect(
            self,
            nodes: List[np.ndarray],
            parents: List[int],
            q_target: np.ndarray,
            mode: PlannerMode,
    ) -> Tuple[str, int, bool]:
        """
        Tries to connect RRT tree given by nodes to a target configuration q_target. We continue extending the tree and
        decide based on status of extension when to stop.

        :param nodes: RRT given in nodes.
        :param parents: RRT nodes that are extended by an edge already.
        :param q_target: Target configuration to which we want to connect given RRT.
        :param mode: Mode of the planner.
        :return: Status of the attempted connection, index of new node in RRT, and collision flag
        """
        collision_any = False

        while True:
            # Extend the tree
            status, new_index, collision_flag = self._extend(nodes, parents, q_target, mode)
            collision_any = collision_any or collision_flag

            # If trapped, we cannot continue extending
            if status == "trapped":
                return "trapped", -1, collision_any
            # If reached, we don't need to extend the tree more
            if status == "reached":
                return "reached", new_index, collision_any
            # If advanced, continue extending

    def _rrt_connect(
            self,
            q_start: np.ndarray,
            q_goal: np.ndarray,
            mode: PlannerMode
    ) -> List[np.ndarray]:
        """
        Alteration to RRT algorithm where there are two trees, tree A and tree B, one rooted in start configuration,
        other in goal configuration. Each iteration we extend both trees, and we extend them until it is not possible or
        until given tree was connected to target configuration.

        We extend tree A first by sampling random target configuration or making the target configuration goal
        configuration. If the extension is stopped by either connecting tree A to target configuration or because the
        extension is no longer possible, we start to extend tree B with target configuration being the last new
        configuration added to the tree A.

        If colliding mode, we return trajectory only if it is a path in tree A, and it leads to colliding configuration.
        If avoidance mode, we return trajectory if both trees share a configuration node or there are configurations
        in the trees connectable by one step.

        :param q_start: Initial configuration of trajectory.
        :param q_goal: Goal configuration of trajectory.
        :param mode: Mode of the planner.
        :return: Trajectory given in joint configurations.
        """

        # Two trees, one rooted in initial configuration node, other in goal configuration node
        nodes_a: List[np.ndarray] = [q_start]
        parents_a: List[int] = [-1]

        nodes_b: List[np.ndarray] = [q_goal]
        parents_b: List[int] = [-1]

        # Track which side is which for correct path direction
        a_is_start = True

        for _ in range(self.cfg.max_rrt_iterations):
            # Sample
            if np.random.random() < self.goal_bias:
                # Bias toward the other root configuration
                q_rand = nodes_b[0]
            else:
                # Pick random configuration
                q_rand = self._sample_joints_angles()

            # Extend tree A toward sample
            status_a, idx_a, collision_a = self._extend(nodes_a, parents_a, q_rand, mode)

            # During collision mode, stop if there is collision and edge is valid meaning extension is not trapped
            if mode == PlannerMode.COLLIDE and collision_a and status_a != "trapped" and a_is_start:
                # Return path from tree A root configuration to colliding configuration if root configuration is initial one
                path_a = _trace_path(nodes_a, parents_a, idx_a)
                return path_a

            # Extension from tree A cannot continue, swap trees and try again
            if status_a == "trapped":
                nodes_a, nodes_b = nodes_b, nodes_a
                parents_a, parents_b = parents_b, parents_a
                a_is_start = not a_is_start
                continue

            q_new = nodes_a[idx_a]

            # Connect tree B toward q_new
            status_b, idx_b, collision_b = self._connect(nodes_b, parents_b, q_new, mode)

            # During collision mode, stop if there is collision and edge is valid meaning extension is not trapped
            if mode == PlannerMode.COLLIDE and collision_b and status_b != "trapped" and not a_is_start:
                # Return path from tree B root configuration to colliding configuration if root configuration is initial one
                path_b = _trace_path(nodes_b, parents_b, idx_b if idx_b != -1 else len(nodes_b) - 1)
                return path_b

            # If connected, build full trajectory
            if status_b == "reached":
                self.env.set_robot_configuration_kinematic(nodes_b[idx_b])
                traj = _merge_paths(nodes_a, parents_a, idx_a, nodes_b, parents_b, idx_b)

                # Ensure direction is start -> goal regardless of swapping
                if not a_is_start:
                    traj.reverse()

                traj = self.smooth_trajectory_through_elastic_band(traj, mode)
                return traj

            # Swap roles
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            a_is_start = not a_is_start

        return []

    def smooth_trajectory_through_removal(
            self,
            trajectory: List[np.ndarray],
            mode: PlannerMode,
    ) -> List[np.ndarray]:
        """
        Used to smooth a trajectory planned by RRT algorithm to make them more humanlike.

        This approach iteratively tests three consecutive joints configurations and tests whether the middle joints
        configuration makes the segment look too jagged. If it does, it is removed from the trajectory following
        restrictions to keep some randomness to the trajectories that are generated.

        :param trajectory: RRT trajectory.
        :param mode: Mode of the planner.
        :return: Smoothed trajectory by removal of joints configurations.
        """
        # Return if there are no nodes to be removed.
        if len(trajectory) <= self.cfg.min_n_nodes_in_path:
            return trajectory

        new_trajectory = trajectory.copy()

        for _ in range(self.cfg.max_passes_during_smoothing):
            if len(new_trajectory) <= self.cfg.min_n_nodes_in_path:
                break

            changed = False
            i = 0

            # Test the three consecutive configurations
            while i <= len(new_trajectory) - 3 and len(new_trajectory) > self.cfg.min_n_nodes_in_path:
                q0, q1, q2 = new_trajectory[i], new_trajectory[i + 1], new_trajectory[i + 2]

                # q1 must create sharp turns and new path q0->q2 must be valid for q1 to be removed
                if _is_jagged_triangle(q0, q1, q2, self.cfg.max_degree_angle_turn):
                    if self._shortcut_valid(q0, q2, mode):
                        new_trajectory.pop(i + 1)
                        changed = True

                        i = max(i - 1, 0)
                        continue

                i += 1

            # If iteration produced no change, so will the next one - end loop
            if not changed:
                break

        return new_trajectory

    def smooth_trajectory_through_elastic_band(
            self,
            trajectory: List[np.ndarray],
            mode: PlannerMode
    )-> List[np.ndarray]:
        """
        Used to smooth a trajectory planned by RRT algorithm to make them more humanlike.

        This approach iteratively takes three consecutive joints configurations and

        :param trajectory: RRT trajectory.
        :param mode: Mode of the planner.
        :return: Smoothed trajectory by movement of joints configurations to more natural configuration
        like an elastic band (Laplacian smoothing).
        """
        # No possible change if trajectory consists of only two configurations
        if len(trajectory) <= 2:
            return trajectory

        # Get joints limits to not produce invalid configurations by pushes
        lows, highs = self.env.robot.joints_limits
        new_trajectory = trajectory.copy()

        for _ in range(self.cfg.max_passes_during_smoothing):
            changed = False

            # Smooth a segment of three consecutive configurations
            for i in range(1, len(new_trajectory) - 1):
                q_prev = new_trajectory[i - 1]
                q = new_trajectory[i]
                q_next = new_trajectory[i + 1]

                # Choose target position based on neighbouring configurations
                q_target = 0.5 * (q_prev + q_next)

                # Calculate change to the middle configuration
                # Control speed of smoothing
                q_target = self._get_next_q_from_action_executable_in_sim_timestep(q, self.cfg.lam * (q_target - q))

                # Change is too small
                if float(np.linalg.norm(q_target - q)) < 1e-6:
                    continue

                # Smoothing operation - find new configuration respecting the joints limits
                q_new = np.clip(q_target, lows, highs)

                # Test if new configuration creates valid paths
                ok1, _ = self._edge_valid(q_prev, q_new, mode)
                if not ok1:
                    continue

                ok2, _ = self._edge_valid(q_new, q_next, mode)
                if not ok2:
                    continue

                # Replace old configuration with the new one if it satisfies conditions
                new_trajectory[i] = q_new
                changed = True

            # If iteration produced no change, so will the next one - end loop
            if not changed:
                break

        return new_trajectory

    def one_euro_filter(
            self,
            trajectory: List[np.ndarray],
            planner_mode: PlannerMode,
            lam_init: float = 1.0,
            lam_decay: float = 0.2,
            lam_min: float = 0.05
    ) -> List[np.ndarray]:
        """
        Smoothing strategy based on one euro filter.

        :param trajectory: Input trajectory.
        :param planner_mode: Mode of the planner.
        :param lam_init: Initial smoothing factor.
        :param lam_decay: Decay of smoothing factor for iterative validation checking.
        :param lam_min: Minimal smoothing factor.
        :return: Smoothed trajectory.
        """

        # No possible change if trajectory consists of only two configurations
        if len(trajectory) <= 2:
            return trajectory

        # Initialize the filter
        one_euro_filter = OneEuroFilter(
            dimension=len(trajectory[0]),
            frequency= 1.0 / self.dt,
            min_cutoff= 1.5,
            beta=0.02,
            derivative_cutoff= 1.0,
        )

        smoothed_trajectory = [trajectory[0].copy()]

        for i in range(1, len(trajectory) - 1):

            q_prev = smoothed_trajectory[-1]
            q = trajectory[i]
            q_next = trajectory[i + 1]

            # Get new q from filter
            q_filtered = one_euro_filter.filter(q)

            # Get new feasible action
            dq = q_filtered - q
            dq = np.clip(dq, -self.dq_max, self.dq_max)

            lam = lam_init

            accepted = False

            # Iteratively check if new edges are valid, if not, decrease the smoothing factor till minimum
            while lam >= lam_min:
                q_candidate = np.clip(q + lam * dq, self.lows, self.highs)

                ok1, _ = self._edge_valid(q_prev, q_candidate, planner_mode)
                if not ok1:
                    lam *= lam_decay
                    continue

                ok2, _ = self._edge_valid(q_candidate, q_next, planner_mode)
                if not ok2:
                    lam *= lam_decay
                    continue

                smoothed_trajectory.append(q_candidate)
                accepted = True
                break

            if not accepted:
                smoothed_trajectory.append(q)

        smoothed_trajectory.append(trajectory[-1])
        return smoothed_trajectory

    def plan(
            self,
            q_start: np.ndarray,
            q_goal: np.ndarray,
            mode: PlannerMode
    ) -> List[np.ndarray]:
        """
        Plans a trajectory based on the selected planner mode using a Rapidly-exploring Random Tree.

        :param q_start: Starting joint configuration.
        :param q_goal: Desired joint configuration that the robot should reach.
        :param mode: Mode of planner.
        :return: Valid trajectory that avoids/collides with collision based on planner mode if one was found.
        """
        # If starting and goal position are practically identical
        if not dist(q_start, q_goal):
            return [q_start.copy()]

        # Try to find a trajectory using RRT
        traj = self._rrt_connect(q_start, q_goal, mode)
        if traj:
            return traj

        return []

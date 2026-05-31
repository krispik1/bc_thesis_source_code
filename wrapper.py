import numpy as np
from typing import Dict, Any, List, Tuple

from dataset_types import State, Action
from geometry_helper import point_to_object_distance, perpendicular_basis, project_point_outside_object, dist
from myGym.envs.env_object import EnvObject
from mygym.myGym.envs.gym_env import GymEnv

def get_action_vector(
        s_t: State,
        s_t1: State,
        desired_joints: np.ndarray
) -> Action:
    """
    Returns an action vector given as the difference between joint angles and status of the robot's magnet of
    state s_t and s_t1.

    :param desired_joints: Desired joint angle configuration.
    :param s_t: Previous state vector.
    :param s_t1: Current state vector.
    :return: Action vector.
    """

    return Action(
        desired_delta_q=desired_joints - s_t.joints_angles,
        delta_q=s_t1.joints_angles - s_t.joints_angles,
        delta_mgt=s_t1.magnet_state - s_t.magnet_state
    )


def filter_unique_configurations(
        configurations: List[np.ndarray],
        threshold: float = 0.1
) -> List[np.ndarray]:
    """
    Filters list of configurations to leave only unique ones that are not within threshold distance of each other.

    :param configurations: Configurations to filter.
    :param threshold: Threshold distance used to filter similar configurations.
    :return: Filtered list of configurations.
    """
    unique = []

    for q in configurations:
        if not any(dist(q, u) < threshold for u in unique):
            unique.append(q)

    return unique


class GymWrapper(GymEnv):

    def __init__(
            self,
            cfg: Dict[str, Any],
            magnet_probability: float,
    ):
        """
        Class that adds additional functionality to the GymEnv class used during data generation.

        :param cfg: Configuration file for the environment.
        """

        # Create base environment based on given config
        super().__init__(
            task_objects=cfg["task_objects"],
            observation=cfg["observation"],
            workspace=cfg["workspace"],
            dimension_velocity=cfg.get("dimension_velocity", 0.05),
            used_objects=cfg["used_objects"],
            action_repeat=cfg["action_repeat"],
            color_dict=cfg.get("color_dict", {}),
            robot=cfg["robot"],
            robot_action=cfg["robot_action"],
            max_velocity=cfg["max_velocity"],
            max_force=cfg["max_force"],
            robot_init_joint_poses=cfg["robot_init"],
            task_type=cfg["task_type"],
            num_networks=cfg.get("num_networks", 1),
            network_switcher=cfg.get("network_switcher", "gt"),
            distractors=cfg["distractors"],

            active_cameras=cfg["camera"],
            dataset=False,
            obs_space=None,
            visualize=cfg["visualize"],
            visgym=cfg["visgym"],
            logdir=cfg["logdir"],

            natural_language=bool(cfg["natural_language"]),
            training=True,
            top_grasp=False,

            gui_on=bool(cfg["gui"]),
            max_ep_steps=cfg["max_episode_steps"],
        )

        # Probability of magnet of the robot turning on
        self.mgt_prob = magnet_probability

        # For memorization of the initiated goal object for resets
        self._goal_init_pos = None
        self._goal_init_orn = None
        self._distractor_init_poses = None
        self.ee_init_pose = self.robot.get_position()

        # IDs of waypoint markers for visualization
        self._waypoint_marker_ids = []

        # Goal poses for planning
        self.goal_poses = []

    def _capture_initial_object_poses(
            self
    ) -> None:
        """
        Called to remember where goal and distractors were
        right after env.reset(), so soft resets can put them back.
        """
        self._goal_init_pos = self.get_goal_position()
        self._goal_init_orn = self.get_goal_orientation()

        distractors = self.task_objects.get("distractor", [])
        self._distractor_init_poses = []
        for d in distractors:
            pos = np.array(d.get_position(), dtype=float)
            orn = np.array(d.get_orientation(), dtype=float)
            self._distractor_init_poses.append((pos, orn))

    def _is_goal_reachable(
            self
    ) -> bool:
        """
        Checks if the goal object is magnetizable by the robotic arm.

        :return: True only if robot was able to pick up the goal object.
        """
        goal_pos = self.get_goal_position().astype(float)

        target_q = self.robot.calculate_accurate_IK(end_effector_pos=goal_pos)
        if target_q is None:
            return False

        self.robot.set_magnetization(1)
        self.step(target_q)

        return len(self.robot.magnetized_objects) > 0

    def _is_distractor_reachable(
            self,
            n_samples: int = 10,
    ) -> bool:
        """
        Checks whether the obstacle is reachable by the robot's arm.

        :param n_samples: Maximum number of sampled points to check.
        :return: True only if the obstacle is reachable (if there is an end-effector position that leads to collision
        with the distractor).
        """

        success = 0

        if not self.get_distractors_positions():
            return True

        for _ in range(n_samples):
            # Sample point on distractor
            sample = self.sample_surface_point_ray()
            if sample is None:
                return False

            point, normal = sample

            # Get end effector position
            ee = point + 0.03 * normal

            # Calculate with IK solver corresponding joint angle configuration
            q = self.robot.calculate_accurate_IK(end_effector_pos=ee)
            if q is None:
                continue

            # Check if the configuration leads to collision
            self.set_robot_configuration_kinematic(q)
            if self.check_robot_distractor_collision():
                success += 1

        return success / n_samples > 0.4


    def reset_until_reachable(
            self,
    ) -> Any:
        """
        Resets the environment until the distractor and the goal object are reachable and there are poses of the arm
        that reach the object without collision with distractor present.

        :return: Observation of the environment.
        """
        while True:
            # Reset env
            self._goal_init_pos = None
            self._goal_init_orn = None
            self._distractor_init_poses = None
            obs = self.reset()

            # Check whether both distractor and goal object are reachable
            if self._is_goal_reachable():
                self.soft_reset_robot_only()
                if not self._is_distractor_reachable():
                   continue

                # Calculate poses and check if they are valid meaning no collision with distractor
                candidate_goal_poses = [self.robot.calculate_accurate_IK(self.get_goal_position())]
                goal_poses = []
                for goal_pose in candidate_goal_poses:
                    if goal_pose is None:
                       continue
                    # Teleport robot to pose and check for collision
                    self.set_robot_configuration_kinematic(goal_pose)
                    if self.check_robot_distractor_collision() or self.check_robot_distractor_penetration()\
                            or not dist(self.get_goal_position(), self.get_ef_position()) < 0.1:
                        continue

                    # If valid, add it to goal poses
                    goal_poses.append(goal_pose)
                # If goal object reachable and there are goal poses, env is usable
                if goal_poses:
                    self._capture_initial_object_poses()
                    self.goal_poses = goal_poses
                    self.soft_reset_robot_only()
                    return obs

    ##########################################
    ### Helper methods for data generation ###
    ##########################################

    def get_ef_position(
            self
    ) -> np.ndarray:
        """
        Used to determine the position of robot's end effector.

        :return: ([x,y,z]) Coordinates of end effector position.
        """

        return np.array(self.robot.get_position(), dtype=float)

    def get_goal_position(
            self
    ) -> np.ndarray:
        """
        Used to determine the position of the goal object.

        :return: ([x, y, z]) Coordinates of goal object.
        """
        goal_obj = self.task_objects["goal_state"]
        goal_obj_pos =  np.array(goal_obj.get_position(), dtype=float)

        return goal_obj_pos

    def get_goal_orientation(
            self
    ) -> np.ndarray:
        """
        Used to determine the orientation of the goal object.

        :return: (quaternion [x,y,z,w]) Orientation of goal object.
        """
        goal_obj = self.task_objects["goal_state"]
        goal_obj_orn = np.array(goal_obj.get_orientation(), dtype=float)

        return goal_obj_orn

    def get_distractors(
            self
    ) -> List[EnvObject]:
        """
        Used to access list of distractors.

        :return: List of EnvObjects representing distractors.
        """
        return self.task_objects.get("distractor", [])

    def get_distractors_positions(
            self
    ) -> List[np.ndarray]:
        """
        Used to determine positions of distractors.

        :return: List of ([x, y, z]) distractors' coordinates.
        """

        return [np.array(distractor.get_position(), dtype=float) for distractor in self.get_distractors()]

    def get_distractors_orientations(
            self
    ) -> List[np.ndarray]:
        """
        Used to determine orientations of distractors.

        :return: List of (quaternion [x,y,z,w]) distractors' orientations.
        """

        return [np.array(distractor.get_orientation(), dtype=float) for distractor in self.get_distractors()]

    def get_distractors_descriptors(
            self
    ) -> List[Tuple[int, List[float]]]:
        """
        Used to get descriptors of distractors depending on their geometrical shape. Geometry shapes match available
        geometry shapes of PyBullet. Based on shape, dimensions are:
            - Sphere - dimensions[0] = radius,
            - Box - dimensions[:3] = extents,
            - Cylinder or Capsule - dimensions[0] = height, dimensions[1] = radius,
            - Mesh - instead of scaling factor, we approximate mesh distractor by sphere and return its radius,
            dimensions[0] = radius of sphere approximating the distractor.

        :return: List of (Geometry of distractor, dimensions based on geometry).
        """

        descriptors: List[Tuple[int, List[float]]] = []

        for distractor in self.get_distractors():
            distractor_collision_data = self.p.getCollisionShapeData(distractor.get_uid(), -1)[0]

            # If mesh, approximate distractor using sphere and calculate its radius using bounding boxes
            if distractor_collision_data[2] == self.p.GEOM_MESH:
                aabb_min, aabb_max = self.p.getAABB(distractor.get_uid())
                aabb_min = np.array(aabb_min, dtype=float)
                aabb_max = np.array(aabb_max, dtype=float)
                extents = aabb_max - aabb_min
                distractor_collision_data[3][0] = [0.5 * np.linalg.norm(extents)]

            descriptors.append((distractor_collision_data[2], distractor_collision_data[3]))

        return descriptors

    def get_distances_end_effector_from_distractors(
            self
    ) -> List[float]:
        """
        Calculates distances between the end effector and distractors.

        :return: List of distances between end effector and distractors.
        """

        distances: List[float] = []
        end_effector_position = self.get_ef_position()

        for descriptor, position, orientation in zip(self.get_distractors_descriptors(), self.get_distractors_positions(), self.get_distractors_orientations()):
            distance = point_to_object_distance(
                point=end_effector_position,
                geometrical_shape=descriptor[0],
                position=position,
                orientation=orientation,
                dimensions=descriptor[1]
            )

            distances.append(distance)

        return distances

    def sample_surface_point_ray(
            self,
            n_rays: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using ray test provided by the physics engine, we sample a point from the true surface of the distractor. For greater
        randomness in sampled rays, we randomly choose one of the faces of axis aligned bounding box for origin point of the
        ray while uniformly sampling other coordinates.

        :param n_rays: Number of rays to use for ray tests.
        :return: Sampled position and normal on true surface of the distractor.
        """
        # Random number generator for choosing faces
        rng = np.random.default_rng()

        # Find centre of aabb
        aabb_min, aabb_max = self.p.getAABB(self.get_distractors()[0].get_uid())
        aabb_min = np.array(aabb_min, dtype=float)
        aabb_max = np.array(aabb_max, dtype=float)
        center = self.get_distractors_positions()[0]

        # Sample rays for ray test batch - ray starts from surface of aabb and ends in centre of aabb/distractor
        rays_from, rays_to = [], []
        for _ in range(n_rays):
            # Pick randomly one of six faces of rectangular block that is aabb
            face = rng.integers(0, 6)

            x = np.random.uniform(aabb_min[0], aabb_max[0])
            y = np.random.uniform(aabb_min[1], aabb_max[1])
            z = np.random.uniform(aabb_min[2], aabb_max[2])

            if face == 0:
                x = aabb_max[0]
            elif face == 1:
                x = aabb_min[0]
            elif face == 2:
                y = aabb_max[1]
            elif face == 3:
                y = aabb_min[1]
            elif face == 4:
                z = aabb_max[2]
            else:
                z = aabb_min[2]

            start = np.array([x, y, z], dtype=float)
            end = center
            rays_from.append(start.tolist())
            rays_to.append(end.tolist())

        results = self.p.rayTestBatch(rays_from, rays_to)

        # Test the results and return position and norm of first suitable ray test result
        for hit_uid, hit_link, hit_frac, hit_pos, hit_norm in results:
            # Ray must hit the distractor
            if hit_uid != self.get_distractors()[0].get_uid() or hit_frac < 0:
                continue

            p_world = np.array(hit_pos, dtype=float)
            n_world = np.array(hit_norm, dtype=float)
            n_world /= max(np.linalg.norm(n_world), 1e-6)
            return p_world, n_world

        # Only if no ray hit the distractor
        raise RuntimeError

    def find_ee_position_for_link_collision(
            self,
            approach_start: float = 0.2,
            approach_end: float = -0.02,
            steps: int = 30
    ) -> np.ndarray:
        """
        As the end effector has no real colliding body, this function is used to find end effector position, using sampled
        point on the surface of the distractor, resulting in collision caused by any link of the robot that is able to collide.

        End effector slowly approaches the sampled point from outside the distractor until a collision occurs or the
        end effector is not located some distance from the sampled point inside the distractor.

        :param approach_start: Starting distance away from the sampled point outside the distractor.
        :param approach_end: Maximum end distance inside the distractor from the sampled point.
        :param steps: Number of steps representing number of nudges.
        :return: End effector position that causes collision.
        """
        # Sample point on surface
        hit_p, hit_n = self.sample_surface_point_ray()

        # Values that move the end effector from start to end through the sampled point, from outside to inside the distractor
        s_values = np.linspace(approach_start, approach_end, steps)

        # Repeatedly nudge the end effector and test for collision
        for s in s_values:
            ee_pos = hit_p + s * hit_n
            q = self.robot.calculate_accurate_IK(end_effector_pos=ee_pos)

            if q is None:
                continue

            self.set_robot_configuration_kinematic(q)

            if self.check_robot_distractor_collision():
                return ee_pos

        # If no end effector position was found
        return self.get_distractors_positions()[0]

    def generate_curved_waypoints_one_obstacle(
            self,
            n_waypoints: int,
            margin: float = 0.05,
            amplitude: float = 0.15,
            randomize_side: bool = True
    ) -> List[np.ndarray]:
        """
        Used to generate waypoints that provide additional randomization of planned trajectories. This approach uses
        sinusoid push of points located on direct path to goal object, meaning chosen number of points will be pushed
        to create curved path. These waypoints are pushed respecting one obstacle environment meaning they will not be
        pushed inside the obstacle which is unreachable for the arm.

        :param n_waypoints: Number of waypoints to generate.
        :param margin: Safety margin to avoid obstacle
        :param amplitude: Amplitude giving curvature to the trajectory.
        :param randomize_side: Whether a side to which points will be pushed is randomized.
        :return: List of 3D coordinates of waypoints.
        """
        start_position = self.get_ef_position()
        goal_position = self.get_goal_position()

        # Represent direct path by vector
        direct_path = goal_position - start_position
        direct_path_length = np.linalg.norm(direct_path)
        # Path is too short
        if direct_path_length < 1e-9:
            return []

        # Get normalized direction vector of direct path and a basis of plane that is orthogonal to that direction
        t_hat = direct_path / (direct_path_length + 1e-9)
        u, w = perpendicular_basis(t_hat)

        # Pick sideways direction
        if randomize_side:
            theta = np.random.uniform(0.0, 2 * np.pi)
            dir_side = np.cos(theta) * u + np.sin(theta) * w
        else:
            dir_side = u

        # Get info about obstacle
        if self.get_distractors():
            descriptor, position, orientation = self.get_distractors_descriptors()[0], self.get_distractors_positions()[0], self.get_distractors_orientations()[0]
        else:
            descriptor, position, orientation = [0, 0], np.zeros(3), np.zeros(4)

        waypoints = []
        for i in range(n_waypoints):
            for j in range(10):
                # Pick anchor (and point to be pushed) on direct path
                t = (i + 1) / (n_waypoints + 1)
                point = start_position + t * direct_path

                # Sinusoid offset
                offset = amplitude * np.sin(np.pi * t)
                point = point + offset * dir_side

                # Point must be outside obstacle
                point = project_point_outside_object(
                    point=point,
                    margin=margin,
                    object_position=position,
                    object_orientation=orientation,
                    object_shape=descriptor[0],
                    object_dimensions=descriptor[1]
                )

                q = self.robot.calculate_accurate_IK(end_effector_pos=point)
                if q is None:
                    continue
                self.set_robot_configuration_kinematic(q)

                if not self.check_robot_distractor_collision():
                    waypoints.append(point)
                    break

        return waypoints

    def calculate_multiple_IK(
            self,
            n_solutions: int = 20
    ) -> List[np.ndarray]:
        """
        Calculates multiple valid IK solutions for end-effector position by randomizing rest poses for IK solver.

        :param n_solutions: Number of solutions we would like to generate,
        :return: Multiple IK solutions.
        """
        solutions = []

        for i in range(n_solutions):
            # Random rest pose inside joint limits
            rest = np.random.uniform(self.robot.joints_limits[0], self.robot.joints_limits[1]).tolist()
            if i == 0:
                rest = None

            joint_poses = self.robot.calculate_accurate_IK(end_effector_pos=self.get_goal_position(), rest=rest)
            if joint_poses is None:
                continue

            solutions.append(joint_poses)

        return solutions

    def set_robot_configuration_kinematic(
            self,
            q: np.ndarray
    ) -> None:
        """
        Teleports robot arm to a desired joint angles configuration. Used to search for candidates in planning and babbling.

        :param q: Joint angles configuration.
        """
        q = np.asarray(q, dtype=float)
        q = np.clip(q, self.robot.joints_limits[0], self.robot.joints_limits[1])
        for jid, idx in enumerate(self.robot.motor_indices):
            self.p.resetJointState(self.robot.robot_uid, idx, float(q[jid]))

        self.p.performCollisionDetection()

    def random_toggle_mgt(
            self
    ) -> None:
        """
        Randomly turns on/off the magnet attached to the arm of the robot based on the given probability.

        Called to randomize the robot's behaviour during dataset generation.
        """
        if np.random.rand() < self.mgt_prob:
            self.robot.set_magnetization(1)
        else:
            self.robot.set_magnetization(0)

    def check_robot_distractor_collision(
            self
    ) -> bool:
        """
        Checks whether the robot collision with the distractors occurs provided by the presence of
        contact points of the robot and the distractors.

        Called during dataset generation to determine colliding states and actions that result in it,
        or collision trajectories.

        :return: True only if a collision has transpired (i.e. there is at least one contact point).
        """

        # Get contact points through the utilized physics engine
        for distractor in self.get_distractors():
            cps = self.p.getContactPoints(self.robot.get_uid(), distractor.get_uid())
            if len(cps) > 0:
                return True

        return False

    def check_robot_distractor_penetration(
            self
    ) -> bool:
        """
        Checks whether the robot glitches through the distractors provided by the presence of negative distance of
        closest points of a distractor to the robot.

        Called during dataset generation to prevent invalid states created by clipping through the distractors by the robot
        as the robot teleports during planning phase.

        :return: True only if the robot is inside a distractor which is invalid state.
        """

        # Get contact points through the utilized physics engine
        for distractor in self.get_distractors():
            points = self.p.getClosestPoints(self.robot.get_uid(), distractor.get_uid(), distance=1e-4)
            if any(point[8] <= 0 for point in points):
                return True

        return False

    def soft_reset_robot_only(
            self
    ) -> Any:
        """
        Soft reset of the environment to generate multiple trajectories with same setup of the objects.

        If it is the first time the environment is going to be reset, we capture the placement of goal object and
        the distractors. Only robot and internal episode representation are reset to initial state. Goal object and
        distractors keep their original placement.

        As the distractors should be unmoveable, we freeze them in place after placing them following the reset.

        :return: Observation of the environment.
        """
        # Memorize initial poses the first time
        if self._goal_init_pos is None:
            self._capture_initial_object_poses()

        # Reset robot state
        self.robot.reset(random_robot=False)
        self.robot.set_magnetization(0)
        if hasattr(self.robot, "release_all_objects"):
            self.robot.release_all_objects()

        # Reset episode
        self.task.reset_task()

        # Put goal object back
        goal_obj = self.task_objects["goal_state"]
        self.p.resetBasePositionAndOrientation(
            goal_obj.get_uid(),
            self._goal_init_pos.tolist(),
            self._goal_init_orn.tolist()
        )

        # Put distractors back
        distractors = self.task_objects.get("distractor", [])
        if self._distractor_init_poses is not None:
            for d, (pos, orn) in zip(distractors, self._distractor_init_poses):
                self.p.resetBasePositionAndOrientation(
                    d.get_uid(),
                    pos.tolist(),
                    orn.tolist()
                )

        # Step once so the engine updates
        self.p.stepSimulation()
        obs = self.get_observation()

        # Freeze distractors in place by reducing their masses to 0
        distractors = self.task_objects.get("distractor", [])
        for d in distractors:
            uid = d.get_uid()
            # link index -1 = base
            self.p.changeDynamics(
                uid,
                -1,
                mass=0.0,  # static
                linearDamping=1.0,
                angularDamping=1.0
            )

        return obs

    ##################################################
    ### Debugging methods for visualization in GUI ###
    ##################################################

    def draw_executed_ee_trajectory_point(
            self,
            prev_pos: np.ndarray,
            curr_pos: np.ndarray,
            line_color: tuple[int, int, int]=(0, 1, 0),
            line_width: float=2.0,
            life_time: float=0.0,
    ) -> None:
        """
        Visualizes the executed movement of the end effector.

        :param prev_pos: Previous position of the end effector.
        :param curr_pos: Current position of the end effector.
        :param line_color: Colour of the line representing the movement.
        :param line_width: Width of the line representing the movement.
        :param life_time: Duration of the visualization. 0.0 -> lines stay until reset/removeAllUserDebugItems.
        """
        self.p.addUserDebugLine(
            prev_pos.tolist(),
            curr_pos.tolist(),
            line_color,
            lineWidth=line_width,
            lifeTime=life_time,
        )

    def show_waypoint_marker(
            self,
            pos: np.ndarray,
            size: float = 0.5,
            color: tuple[int, int, int]=(1, 1, 0),
            life_time: float=0.0,
    ) -> None:
        """
        Visualizes the waypoint marker of planned trajectory.

        :param pos: Position of the waypoint marker.
        :param size: Size of the marker.
        :param color: Colour of the marker.
        :param life_time: Duration of the visualization. 0.0 -> lines stay until reset/removeAllUserDebugItems.
        """
        x, y, z = pos.tolist()
        # Lines creating the cross representation of the marker
        lines = [
            # X-axis
            self.p.addUserDebugLine(
                [x - size, y, z],
                [x + size, y, z],
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=life_time,
            ),
            # Y-axis
            self.p.addUserDebugLine(
                [x, y - size, z],
                [x, y + size, z],
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=life_time,
            ),
            # Z-axis
            self.p.addUserDebugLine(
                [x, y, z - size],
                [x, y, z + size],
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=life_time,
            )
        ]

        #Remember new marker for selective removal
        self._waypoint_marker_ids.extend(lines)

    def clear_waypoint_marker(
            self
    ) -> None:
        """
        Clears waypoint markers based on the waypoint marker ids.

        Clears memorized waypoint markers.
        """
        for uid in self._waypoint_marker_ids:
            self.p.removeUserDebugItem(uid)

        self._waypoint_marker_ids = []

    def draw_box(
            self,
            bounds: list[float],
            color: tuple[int, int, int]=(1, 0, 0),
            line_width: float=2.0,
            life_time: float=0.0,
    ) -> None:
        """
        Draw an axis-aligned bounding box using debug lines.

        :param bounds: ([x_min, x_max, y_min, y_max, z_min, z_max]) Geometry of the bounding box given through bounds.
        :param color: Color of the bounding box.
        :param line_width: Width of lines of the bounding box.
        :param life_time: Duration of the visualization. 0.0 -> lines stay until reset/removeAllUserDebugItems
        """
        # Bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        # Corners of the box
        p000 = [x_min, y_min, z_min]
        p001 = [x_min, y_min, z_max]
        p010 = [x_min, y_max, z_min]
        p011 = [x_min, y_max, z_max]
        p100 = [x_max, y_min, z_min]
        p101 = [x_max, y_min, z_max]
        p110 = [x_max, y_max, z_min]
        p111 = [x_max, y_max, z_max]

        points = [p000, p001, p010, p011, p100, p101, p110, p111]

        # Edges of the box as (start_index, end_index) in points
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]

        # Draw the box using its edges as debug lines
        for i, j in edges:
            self.p.addUserDebugLine(
                points[i],
                points[j],
                color,
                lineWidth=line_width,
                lifeTime=life_time,
            )

    ################################################################
    ### State and action representation methods for logging data ###
    ################################################################

    def get_state_vector(
            self
    ) -> State:
        """
        Based on the observation of the environment, returns a state vector containing:
            - joint angles of the robot
            - end effector description as position and rotation
            - goal object description as position and rotation
            - obstacles (distractors) present in the environment and their description given as position and rotation
            - state of the magnet (bool)

        :return: State vector.
        """
        state = self.get_observation()

        return State(
            joints_angles=np.asarray(state["additional_obs"]["joints_angles"], dtype=float),
            end_effector6D=np.asarray(state["actual_state"], dtype=float),
            goal_object6D=np.asarray(state["goal_state"], dtype=float),
            obstacle6D=np.asarray(
                state["additional_obs"].get("distractor", []),
                dtype=float
            ),
            magnet_state=self.robot.use_magnet
        )

from typing import List, Dict, Tuple, Optional

import numpy as np

from remake.dataset_types import Region
from remake.geometry_helper import dist
from remake.wrapper import GymWrapper


def _neighbouring_keys(
        key: Tuple[int, ...],
) -> List[Tuple[int, ...]]:
    """
    Finds neighbouring cell keys of the hash grid for given key.

    :param key: Cell key to a cell in hash grid.
    :return: List of neighbouring cell keys.
    """

    D = len(key)
    offsets = np.array(np.meshgrid(*([[-1, 0, 1]] * D))).T.reshape(-1, D)
    base = np.array(key, dtype=int)
    return [tuple((base + o).tolist()) for o in offsets]


class PoissonSampler:

    def __init__(
            self,
            env: GymWrapper,
            d_near: float,
            d_contact: float,
            q_min: np.ndarray = np.array([-2.967, -1.833, -2.967, -3.142, -2.967, -0.087, -2.967]),
            q_max: np.ndarray = np.array([2.967, 1.833, 2.967, 0.0, 2.967, 3.822, 2.967]),
            r: float = 0.10
    ):
        """
        Class responsible for sampling initial configurations of robot using Poisson Disk sampling to ensure better
        coverage of environment in the generated dataset.

        Environment is divided into 3 regions: far, near and contact.

        :param env: Environment to sample from.
        :param d_near: Boundary between far and near region.
        :param d_contact: Boundary between near and contact region.
        :param q_min: Maximum for sampled configuration.
        :param q_max: Minimum for sampled configuration.
        :param r: Poisson disk distance or how far new sample should be located away from other samples.
        """
        self.q_min = q_min
        self.q_max = q_max

        self.env = env
        self.lows: np.ndarray = np.asarray(env.robot.joints_limits[0], dtype=np.float64)
        self.highs: np.ndarray = np.asarray(env.robot.joints_limits[1], dtype=np.float64)

        self.d_near = d_near
        self.d_contact = d_contact

        # To ensure Poisson disk condition, we use hash grid
        self.cell_size = r
        self.grid: Dict[Tuple[int, ...], List[np.ndarray]] = {}

    def _normalize_q(
            self,
            q: np.ndarray,
    ) -> np.ndarray:
        """
        Normalizes configuration to be in interval [0, 1].

        :param q: Configuration to be normalized.
        :return: Normalized configuration.
        """
        span = np.maximum(self.highs - self.lows, 1e-9)

        return (q - self.lows) / span

    def _cell_key(
            self,
            x: np.ndarray,
    ) -> Tuple[int, ...]:
        """
        Finds cell key used in hash grid for given configuration.

        :param x: Configuration.
        :return: Key of the cell configuration is in.
        """

        return tuple(np.floor(x / self.cell_size).astype(int))

    def _add(
            self,
            q: np.ndarray,
    ) -> None:
        """
        Adds new configuration to grid.

        :param q: Configuration to be added.
        """

        x = self._normalize_q(q)
        k = self._cell_key(x)

        self.grid.setdefault(k, []).append(q)

    def _is_far_enough(
            self,
            q: np.ndarray,
    ) -> bool:
        """
        Ensures poisson disk condition. Checks only neighbouring cells of hash grid as it is enough to ensure that
        the configuration is at least r distance from other initial configurations.
        :param q:
        :return:
        """
        x = self._normalize_q(q)
        k = self._cell_key(x)

        for nk in _neighbouring_keys(k):
            for qs in self.grid.get(nk, []):
                xs = self._normalize_q(qs)
                if dist(x, xs) < self.cell_size:
                    return False

        return True

    def sample(
            self,
            region: Region,
            max_tries: int = 2000,
    ) -> Optional[np.ndarray]:
        """
        Samples initial configuration so it is at least r distance away from other initial configurations.

        :param region: Region the configuration should be sampled from.
        :param max_tries: Maximum number of tries to sample.
        :return: Initial configuration in the region, if one was found.
        """
        # If contact region, find initial pose that is near enough for colliding step
        if region == Region.CONTACT:
            for _ in range(300):
                # Try to get end effector close to the obstacle
                q_seed = np.random.uniform(self.q_min, self.q_max)
                self.env.set_robot_configuration_kinematic(q_seed)
                distance = self.env.get_distances_end_effector_from_distractors()[0]

                # If end effector position is too far away, skip
                if distance >= max(self.d_near, 0.15):
                    continue

                for _ in range(80):
                    # Nudge the end effector close
                    delta_q = np.random.normal(0.0, 0.10, size=q_seed.shape)
                    q = np.clip(q_seed + delta_q, self.lows, self.highs)

                    self.env.set_robot_configuration_kinematic(q)
                    distance_from_obstacle = self.env.get_distances_end_effector_from_distractors()[0]

                    # If it is close enough, use it as sample
                    if distance_from_obstacle < self.d_contact:
                        self._add(q)
                        return q

            return None

        for _ in range(max_tries):
            # Sample a configuration
            q = np.random.uniform(self.q_min, self.q_max)

            # Teleport arm to that configuration
            self.env.set_robot_configuration_kinematic(q)

            # If the region does not matter, use it as initial configuration
            if region == Region.ANY:
                self._add(q)
                return q

            # Check if it is far enough from the obstacle
            distance_from_obstacle = self.env.get_distances_end_effector_from_distractors()[0]

            # Above boundary for far region
            if region == Region.FAR and distance_from_obstacle < self.d_near:
                continue
            # Bellow boundary for near region
            if region == Region.NEAR and distance_from_obstacle >= self.d_near:
                continue
            # End effector in reachable distance of obstacle for robot (keep in mind effector has no collision shape)
            if region == Region.CONTACT and distance_from_obstacle >= self.d_contact:
                continue

            # Ensure poisson disk condition for far and near region, but ignore for contact
            if region == Region.CONTACT or self._is_far_enough(q):
                self._add(q)
                return q

        return None
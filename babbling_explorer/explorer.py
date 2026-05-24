from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from remake.config import TrainingDatasetGenerationConfig
from remake.dataset_types import Action, Episode, Region, Transition, PlannerPolicy, PlannerMode
from remake.babbling_explorer.poisson_sampler import PoissonSampler
from remake.babbling_explorer.proxy_ensemble import ProxyEnsemble,  flatten_state
from remake.babbling_explorer.replay_buffer import ReplayBuffer
from remake.geometry_helper import dist
from remake.wrapper import GymWrapper, get_action_vector

def sample_unit_vector(
        dimension: int,
) -> np.ndarray:
    """
    Samples unit vector.

    :param dimension: Dimension of the vector.
    :return: Unit vector with given dimension.
    """
    v = np.random.normal(size=dimension)
    n = np.linalg.norm(v) + 1e-12
    return v / n

class Explorer:

    def __init__(
            self,
            env: GymWrapper,
            cfg: TrainingDatasetGenerationConfig,
            device: str = 'cpu',
    ):
        """
        Class generating babbling data that samples candidates for action and uses exploration via disagreement to
        provide better coverage of the environment in generated data. Every few steps a new initial position is picked
        to babble from.

        Initial position of robot is provided by Poisson Disk sampler to start to avoid areas that are too dense in
        initial configurations.

        Disagreement is provided by a proxy ensemble of multi-layered perceptron. Each perceptron predicts an output
        based on same candidate input and disagreement stems from different predictions. The MLPs are also trained to improve
        predictions as they get more familiar with some parts of the environment. This means their disagreement should
        point to uncertainty, and so we pick a candidate with the highest disagreement which should lead us to more
        uncertain regions.

        :param env: Environment.
        :param cfg: Config file.
        :param device: Device.
        """
        self.env = env
        self.cfg = cfg
        self.observation_counter = 0

        # Joint limits provide sample space for configurations and actions dq
        self.lows, self.highs = self.env.robot.joints_limits

        # Action space is given by maximum velocity and time step of environment step - max difference in configurations
        # for one step must be executable in one environment step
        dt = self.env.p.getPhysicsEngineParameters()["fixedTimeStep"]
        v_max = np.asarray(self.env.robot.joints_max_velo, dtype=float)
        # First float = safety clearance
        self.dq_max = 0.5 * self.cfg.env_config["action_repeat"] * v_max * dt

        # Bookkeeping for region ratio
        self.region_counts = [0, 0, 0]
        self.target_region_ratio = np.array(self.cfg.region_ratio)

        # Settings for MLPs
        state_dimension = len(flatten_state(self.env.get_state_vector()))
        action_dimension = len(self.env.robot.motor_indices) + 1
        input_dimension = state_dimension + action_dimension
        output_dimension = state_dimension

        self.buffer = ReplayBuffer(self.cfg.replay_capacity)

        self.initial_position_sampler = PoissonSampler(
            env=self.env,
            q_min=np.array(self.cfg.q_min),
            q_max=np.array(self.cfg.q_max),
            d_near=self.cfg.d_near,
            d_contact=self.cfg.d_contact,
            r=self.cfg.r
        )

        self.proxy_ensemble = ProxyEnsemble(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=self.cfg.hidden,
            n_models=self.cfg.n_models,
            lr=self.cfg.lr,
            device=device,
        )

        # Data structure for generated contact transitions not selected yet
        self.mem_contact_transitions = deque(maxlen=1000)

        # Obstacle flag
        self.no_obstacle = not bool(cfg.obstacle_present)

    def _choose_region_based_on_representation_ratio(
            self,
            total_observations: int,
    ) -> Region:
        """
        Chooses region that is underrepresented in the dataset.

        :param total_observations: Total number of observations.
        :return: Underrepresented region.
        """

        # If no obstacle, use no division of regions
        if self.no_obstacle:
            return Region.ANY

        # If no observations, choose far
        if not total_observations:
            return Region.FAR

        current_ratio = np.array(self.region_counts) / total_observations
        deficit = self.target_region_ratio - current_ratio

        return Region(int(np.argmax(deficit)))

    def _sample_start_configuration_in_sampled_region(
            self,
            region: Region,
    ) -> bool:
        """
        Based on region, sample initial configuration using Poisson sampler and set robot to it.

        :param region: Region in which the end effector during initial configuration should be.
        :return: True only if robot is successfully set.
        """
        # Find initial configuration, if no configuration is found, abort
        initial_configuration = self.initial_position_sampler.sample(region)
        if initial_configuration is None:
            return False

        # Set robot to initial configuration
        self.env.set_robot_configuration_kinematic(initial_configuration)

        # If robot is inside obstacle in initial position, abort
        if self.env.check_robot_distractor_collision() or self.env.check_robot_distractor_penetration():
            return False

        return True

    def _generate_free_candidates(
            self
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate collision-free candidate configurations that will be used for next step, leading to new state and
        creating new observation.

        :return: List of pairs of start configuration collision-free candidate configuration.
        """
        candidates : List[Tuple[np.ndarray, np.ndarray]] = []
        q_start = self.env.robot.get_joints_states()

        # Scales for step size
        scales = [1.0, 0.75, 0.5, 0.25]

        for _ in range(self.cfg.n_candidates):
            # Scale direction vector to try to find next action
            for s in scales:
                # Sample an action scaled and clipped it so it respects joint limits
                delta_q = np.random.uniform(-self.dq_max * s, self.dq_max * s)
                q_candidate = np.clip(q_start + delta_q, self.lows, self.highs)

                # Teleport robot to candidate, check for collision
                self.env.set_robot_configuration_kinematic(q_candidate)

                # Allow candidate only if no collision happens
                if not self.env.check_robot_distractor_collision() and not self.env.check_robot_distractor_penetration():
                    candidates.append((q_start, q_candidate))
                    break

        return candidates

    def _generate_contact_transition_candidates(
            self,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates candidate pairs for collision step that generates observation/transition with collision.

        :return: List of pairs of start configuration and colliding candidate configuration.
        """

        candidates: List[Tuple[np.ndarray, np.ndarray]] = []

        # Use not yet selected candidates
        while self.mem_contact_transitions and len(candidates) < self.cfg.n_candidates:
            candidates.append(self.mem_contact_transitions.popleft())

        if self.cfg.n_candidates <= len(candidates):
            return candidates

        # Scale backing away from obstacle
        backoff_scales = [1.0, 0.75, 0.5, 0.25]

        for _ in range(300):
            # Get colliding configuration
            q_coll = self.env.robot.calculate_accurate_IK(self.env.find_ee_position_for_link_collision())
            if q_coll is None:
                continue

            self.env.set_robot_configuration_kinematic(q_coll)
            if not self.env.check_robot_distractor_collision():
                continue

            # Try to find numerous start configurations that are non-colliding
            for _ in range(50):
                direction = sample_unit_vector(q_coll.shape[0])
                added = False
                for scale in backoff_scales:
                    # Try to back off respecting joint limits
                    dq = np.clip(direction * scale, -self.dq_max, self.dq_max)
                    q_free = np.clip(q_coll - dq, self.lows, self.highs)
                    self.env.set_robot_configuration_kinematic(q_free)
                    if self.env.check_robot_distractor_collision():
                        continue
                    if np.any(np.abs(q_free - q_coll) > self.dq_max):
                        continue

                    candidates.append((q_free, q_coll))
                    added = True
                    break

                if len(candidates) >= self.cfg.n_candidates or added:
                    break

            if len(candidates) >= self.cfg.n_candidates:
                break

        return candidates

    def _is_transition_valid(
            self,
            sampled_region: Region,
    ) -> bool:
        """
        Checks if transition is valid based on sampled region.

        In contact region, transition is valid only if it is colliding. In far/near region, it is valid only if there is
        no collision.

        :param sampled_region: Sampled region.
        :return: True only if transition is valid.
        """

        return (sampled_region == Region.CONTACT) == (self.env.check_robot_distractor_collision())

    def _get_max_uncertainty_transition(
            self,
            candidates: List[Tuple[np.ndarray, np.ndarray]],
            cache: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets transition which has the highest uncertainty decided by the proxy ensemble.

        :param candidates: List of candidate pairs of start configuration and end configuration.
        :param cache: Cache flag - whether not selected candidates should be cached.
        :return: Start configuration and end configuration that has the highest uncertainty.
        """

        max_uncertainty_transition = None
        max_uncertainty = -np.inf

        for q_start, q_end in candidates:
            self.env.set_robot_configuration_kinematic(q_start)

            state_vector = self.env.get_state_vector()
            action = Action(q_end - q_start, q_end - q_start, np.random.randint(-1, 2))

            # Get uncertainty for each candidate
            uncertainty = self.proxy_ensemble.disagreement(state_vector, action) - 0.1 * dist(q_start, q_end)

            # If the highest so far, update the current highest and cache
            if uncertainty > max_uncertainty:
                if cache and max_uncertainty_transition is not None:
                    self.mem_contact_transitions.append(max_uncertainty_transition)
                max_uncertainty = uncertainty
                max_uncertainty_transition = (q_start, q_end)
                continue

            # Cache if not used
            if cache:
                self.mem_contact_transitions.append((q_start, q_end))


        return max_uncertainty_transition

    def _get_transitions(
            self,
            sampled_region: Region,
            n_steps: int,
    ) -> List[Transition]:
        """
        Runs episode and gets transitions given a sampled region.

        :param sampled_region: Sampled region from which to generate transitions.
        :param n_steps: Number of steps to generate.
        :return: List of transitions or empty list if none were generated.
        """

        transitions: List[Transition] = []

        # If region is not contact, sample starting position for this episode
        if not sampled_region == Region.CONTACT and not self._sample_start_configuration_in_sampled_region(sampled_region):
            return []

        # To prevent bottleneck
        step_counter = 0
        while len(transitions) < n_steps and step_counter < 500:
            step_counter += 1

            # Sample candidates based on given region, if none were found, continue
            if sampled_region == Region.CONTACT:
                candidates = self._generate_contact_transition_candidates()
            else:
                candidates = self._generate_free_candidates()
            if not candidates:
                continue

            # Get start and end configuration of most uncertain transition
            q_from, q_to = self._get_max_uncertainty_transition(candidates, (sampled_region == Region.CONTACT))

            # Perform step from original configuration to maximum uncertainty configuration
            self.env.set_robot_configuration_kinematic(q_from)
            state_vector = self.env.get_state_vector()

            self.env.step(q_to)

            # Check if transition is valid
            if not self._is_transition_valid(sampled_region):
                continue

            new_state_vector = self.env.get_state_vector()

            transition = Transition(
                state_t=state_vector,
                action=get_action_vector(state_vector, new_state_vector, q_to - q_from),
                state_t1=new_state_vector,
                step_collision=(sampled_region == Region.CONTACT),
            )

            # Add transition to replay buffer
            self.buffer.add(transition)

            # Add to region counts and observation count or skip the transition if not respecting collision rules
            if not sampled_region == Region.ANY:
                self.region_counts[sampled_region] += 1

            self.observation_counter += 1

            transitions.append(transition)

        return transitions

    def update_env(
            self
    ) -> None:
        """
        Updates env.
        """

        self.initial_position_sampler.env = self.env

    def collect_data(
            self,
            mode: str
    ) -> Optional[Episode]:
        """
        Collects babbling data. These transitions are represented by an episode. Each episode has randomly sampled
        number of transitions and starts in new initial configuration.

        :param mode: Mode of collection - next means change the environment.
        :return: Babbling episode if one was finished.
        """

        # If next, update env and clear cache
        if mode == "next":
            self.mem_contact_transitions.clear()
            self.env.reset_until_reachable()
            self.update_env()
        else:
            self.env.soft_reset_robot_only()

        # Sample underrepresented region
        sampled_region = self._choose_region_based_on_representation_ratio(self.observation_counter)

        # Sample number of steps which are doubled if we want collision
        n_steps = int(np.random.randint(
            self.cfg.interval_n_steps_per_exploration[0],
            self.cfg.interval_n_steps_per_exploration[1])
        )

        # Collections are split if contact/free region
        transitions = self._get_transitions(sampled_region, n_steps)

        # Train ensemble if there are enough samples
        if len(self.buffer) > self.cfg.batch_size:
            self.proxy_ensemble.train(
                batch=self.buffer.sample(self.cfg.batch_size),
                steps=self.cfg.train_steps_per_env_step
            )

        # If there are no transitions, no episode was run
        if not transitions:
            return None

        # Add planner mode based on region of babbling
        if sampled_region == Region.CONTACT:
            planner_mode = PlannerMode.COLLIDE
        else:
            planner_mode = PlannerMode.AVOID

        return Episode(
            transitions=transitions,
            episode_collision=(sampled_region == Region.CONTACT),
            planner_policy=PlannerPolicy.BABBLING,
            planner_mode=planner_mode,
            success=False,
        )
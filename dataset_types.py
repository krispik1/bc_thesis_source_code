from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from typing import List

class RobotName(IntEnum):
    """
    Integer enum representing the robot name.
    """
    KUKA = 0

ROBOT_NAME_FROM_STR = {
    "kuka": RobotName.KUKA,
}

ROBOT_NAME_TO_STR = {v: k for k, v in ROBOT_NAME_FROM_STR.items()}

class RobotAction(IntEnum):
    """
    Integer enum representing the robot action format.
    """
    JOINTS = 0

ROBOT_ACTION_FROM_STR = {
    "joints": RobotAction.JOINTS,
}

ROBOT_ACTION_TO_STR = {v: k for k, v in ROBOT_ACTION_FROM_STR.items()}

class GoalObject(IntEnum):
    """
    Integer enum representing the goal object name.
    """
    CUBE_HOLES = 0

GOAL_OBJECT_FROM_STR = {
    "cube_holes": GoalObject.CUBE_HOLES,
}

GOAL_OBJECT_TO_STR = {v: k for k, v in GOAL_OBJECT_FROM_STR.items()}

class Obstacle(IntEnum):
    """
    Integer enum representing the obstacle object name.
    """
    CUBOID = 0

OBSTACLE_FROM_STR = {
    "cuboid": Obstacle.CUBOID,
}

OBSTACLE_TO_STR = {v: k for k, v in OBSTACLE_FROM_STR.items()}

class PlannerMode(IntEnum):
    """
    Integer enum representing the planner mode:
        - DIRECT - plans direct trajectory to the goal object,
        - AVOID - plans trajectory avoiding the obstacle to the goal object,
        - COLLIDE - plans trajectory colliding with the obstacle.
    """
    DIRECT = 0
    AVOID = 1
    COLLIDE = 2

PLANNER_MODE_FROM_STR = {
    "direct": PlannerMode.DIRECT,
    "avoid": PlannerMode.AVOID,
    "collide": PlannerMode.COLLIDE,
}

PLANNER_MODE_TO_STR = {v: k for k, v in PLANNER_MODE_FROM_STR.items()}

class PlannerPolicy(IntEnum):
    """
    Integer enum representing the planner's policy:
        - BABBLING - plans a trajectory representing motor babbling in the environment,
        - WAYPOINT - plans a trajectory represented by waypoints which avoids the obstacle through detour waypoint
        or collides with the obstacle in impact point,
        - RRT - plans a trajectory using Rapidly-exploring Random Trees,
        - RRT_WAYPOINT - plans a trajectory by using RRT algorithm for segments represented by waypoints provided by
        WAYPOINT policy.
    """
    BABBLING = 0
    RRT = 1

PLANNER_POLICY_FROM_STR = {
    "babbling": PlannerPolicy.BABBLING,
    "rrt": PlannerPolicy.RRT,
}

PLANNER_POLICY_TO_STR = {v: k for k, v in PLANNER_POLICY_FROM_STR.items()}

@dataclass
class State:
    """
    Dataclass representing a state vector:
        - joints_angles (n_joints, ) - angles of the robot's joints,
        - end_effector6D (7, ) - 6D pose of the robot's end effector,
        - goal_object6D (7, ) - 6D pose of the goal object,
        - obstacle6D (7, ) - 6D pose of the obstacle,
        - magnet_state (0 = Off/1 = On) - state of the magnet.
    """
    joints_angles: np.ndarray
    end_effector6D: np.ndarray
    goal_object6D: np.ndarray
    obstacle6D: np.ndarray
    magnet_state: int

@dataclass
class Action:
    """
    Dataclass representing an action vector:
        - desired_delta_q (n_joints, ) - difference of joint angles in current and desired state
        - delta_q (n_joints, ) - difference of joints angles of two states,
        - delta_mgt (-1/0/1) - difference of magnet states of two states.
    """
    desired_delta_q: np.ndarray
    delta_q: np.ndarray
    delta_mgt: int

@dataclass
class Transition:
    """
    Dataclass representing a transition between two states:
        - state_t (State) - previous state vector,
        - state_t1 (State) - new state vector,
        - action (Action) - action vector resulting in transition between state_t and state_t1,
        - step_collision (bool) - whether collision occurred or not in the transition.
    """
    state_t: State
    action: Action
    state_t1: State
    step_collision: bool

@dataclass
class Episode:
    """
    Dataclass representing an episode:
        - transition (List[Transition]) - list of transitions representing the episode,
        - episode_collision (bool) - whether collision occurred or not in the episode,
        - planner_policy (PlannerPolicy) - planner policy utilized for planning the episode's trajectory,
        - planner_mode (PlannerMode) - planner mode (direct/avoid/collide) utilized for planning the episode's trajectory,
        - success (bool) - whether the robot successfully magnetizes the goal object.
    """
    transitions: List[Transition]
    episode_collision: bool
    planner_policy: PlannerPolicy
    planner_mode: PlannerMode
    success: bool

class Region(IntEnum):
    """
    Dataclass representing a region from region division defined in config file for explorer:
        - FAR - region far away from the obstacle,
        - NEAR - region near  the obstacle,
        - CONTACT - represents contact transitions - boundary region of the obstacle,
        - ANY - region division is not needed - used for env with no obstacle.
    """
    FAR = 0
    NEAR = 1
    CONTACT = 2
    ANY = 3

REGION_FROM_STR = {
    "far": Region.FAR,
    "near": Region.NEAR,
    "contact": Region.CONTACT,
}

REGION_TO_STR = {v: k for k, v in REGION_FROM_STR.items()}
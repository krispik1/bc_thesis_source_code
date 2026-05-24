from dataclasses import dataclass

import numpy as np

@dataclass
class ColumnSpecification:
    """
    Dataclass specifying column attributes:
        - dtype - type of data
        - shape - dimension of data
    """
    dtype: np.dtype
    shape: tuple[int, ...] = ()


class Schema:
    """
    Class specifying schema attributes:
        - name - name of schema
        - column_names - list of column names
        - column_specifications - list of column specifications
        - n_cols - number of columns
    """
    name: str
    column_names: list[str]
    column_specifications: list[ColumnSpecification]

    def __init__(
            self,
            name: str,
            column_names: list[str],
            column_specifications: list[ColumnSpecification]
    ) -> None:
        # The number of names and specifications for columns must be equal
        assert len(column_names) == len(column_specifications)

        self.name = name
        self.column_names = column_names
        self.n_cols = len(column_names)
        self.column_specifications = column_specifications

# Schema of table containing parameters of used environments
ENV_SCHEMA = Schema(
    name='env',
    column_names=[
        'robot_name',
        'robot_action',
        'robot_init',
        'goal_obj_name',
        'goal_obj6D',
        'obstacle_name',
        'obstacle6D',
    ],
    column_specifications=[
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(3,)
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
    ]
)

# Schema of a table containing transitions (s_t, a_t, s_t1) of run episodes
TRANSITION_SCHEMA = Schema(
    name='transitions',
    column_names=[
        'joints_angles_t',
        'ee6D_t',
        'goal_obj6D_t',
        'obstacle6D_t',
        'mgt_t',
        'joints_angles_t1',
        'ee6D_t1',
        'goal_obj6D_t1',
        'obstacle6D_t1',
        'mgt_t1',
        'desired_delta_q',
        'delta_q',
        'delta_mgt',
        'collision',
    ],
    column_specifications=[
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.int8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.int8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.float64,
            shape=(7,)
        ),
        ColumnSpecification(
            dtype=np.int8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.int8,
            shape=()
        ),
    ]
)

# Schema of a table containing information about run episodes
# env_id is the index of the environment used for the episode in env table
# transitions of the episode are [ep_start: ep_start + ep_len] elements in transitions table
EPISODE_SCHEMA = Schema(
    name='episodes',
    column_names=[
        'env_id',
        'ep_start',
        'ep_len',
        'planner',
        'mode',
        'collision',
        'success',
    ],
    column_specifications=[
        ColumnSpecification(
            dtype=np.uint64,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint64,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint64,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        ),
        ColumnSpecification(
            dtype=np.uint8,
            shape=()
        )
    ]
)

SCHEMAS = [ENV_SCHEMA, TRANSITION_SCHEMA, EPISODE_SCHEMA]

# Schema dictionaries
SCHEMA_FROM_STR = {
    "env": ENV_SCHEMA,
    "transition": TRANSITION_SCHEMA,
    "episode": EPISODE_SCHEMA,
}

SCHEMA_TO_STR = {v: k for k, v in SCHEMA_FROM_STR.items()}
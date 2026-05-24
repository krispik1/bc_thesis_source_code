from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class PretrainTrajectoryModelDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        datapoint_indices: List[int],
    ):
        """
        Trajectory model training and validation dataset, optionally used for supervised training and pretraining.
        Adjusted for parallel reading.

        :param h5_path: Path to h5 file with trajectory data.
        :param datapoint_indices: Indices of datapoints in trajectory dataset.
        """

        self.h5_path = h5_path
        self.datapoint_indices = datapoint_indices
        self.h5_file = None

        # Horizon and heads of the model
        self.n_timesteps = 50
        self.trajectory_state = {
            "configuration": 7,
            "end_effector_position": 3,
            "end_effector_rotation": 4,
            "magnet": 1,
            "goal_obj6D_position": 3,
            "goal_obj6D_rotation": 4,
            "obstacle6D_position": 3,
            "obstacle6D_rotation": 4
        }

    def __getstate__(
            self
    ):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __setstate__(
            self,
            state
    ):
        self.__dict__.update(state)
        self.h5_file = None

    def _get_h5_file(
            self
    ) -> h5py.File:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.h5_file

    def __len__(
            self
    ) -> int:
        return len(self.datapoint_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        datapoint_idx = self.datapoint_indices[idx]

        h5_file = self._get_h5_file()
        episodes = h5_file["episodes"]
        transitions = h5_file["transitions"]

        episode_start = episodes["ep_start"][datapoint_idx]
        episode_len = episodes["ep_len"][datapoint_idx]

        initial_transition_idx = episode_start
        final_transition_idx = int(episode_start + episode_len - 1)

        x = torch.from_numpy(
            np.concatenate(
                (
                    transitions["joints_angles_t"][initial_transition_idx],
                    transitions["ee6D_t"][initial_transition_idx],
                    np.atleast_1d(transitions["mgt_t"][initial_transition_idx]),
                    transitions["goal_obj6D_t"][initial_transition_idx],
                    transitions["obstacle6D_t"][initial_transition_idx],
                    transitions["joints_angles_t"][final_transition_idx],
                    transitions["ee6D_t"][final_transition_idx],
                    np.atleast_1d(transitions["mgt_t"][final_transition_idx]),
                    transitions["goal_obj6D_t"][final_transition_idx],
                    transitions["obstacle6D_t"][final_transition_idx],
                )
            )
        ).float()

        initial_state_configuration = torch.from_numpy(
            transitions["joints_angles_t"][initial_transition_idx]
        ).float()
        initial_state_end_effector = torch.from_numpy(
            transitions["ee6D_t"][initial_transition_idx]
        ).float()
        initial_state_mgt = torch.from_numpy(
            np.atleast_1d(transitions["mgt_t"][initial_transition_idx]),
        ).float()
        initial_state_goal_obj6D = torch.from_numpy(
            transitions["goal_obj6D_t"][initial_transition_idx]
        ).float()
        initial_state_obstacle6D = torch.from_numpy(
            transitions["obstacle6D_t"][initial_transition_idx]
        ).float()

        final_state_configuration = torch.from_numpy(
            transitions["joints_angles_t"][final_transition_idx]
        ).float()
        final_state_end_effector = torch.from_numpy(
            transitions["ee6D_t"][final_transition_idx]
        ).float()
        final_state_mgt = torch.from_numpy(
            np.atleast_1d(transitions["mgt_t"][final_transition_idx]),
        ).float()
        final_state_goal_obj6D = torch.from_numpy(
            transitions["goal_obj6D_t"][final_transition_idx]
        ).float()
        final_state_obstacle6D = torch.from_numpy(
            transitions["obstacle6D_t"][final_transition_idx]
        ).float()

        target_trajectory: Dict[str, torch.Tensor] = {
            output_name: torch.zeros(self.n_timesteps, output_dimension)
            for output_name, output_dimension in self.trajectory_state.items()
        }

        for i in range(self.n_timesteps):
            relative_idx = i + 1

            if relative_idx >= episode_len:
                relative_idx = episode_len - 1

            idx = int(episode_start + relative_idx)

            target_trajectory["configuration"][i] = torch.from_numpy(
                transitions["joints_angles_t"][idx]
            ).float()
            target_trajectory["end_effector_position"][i] = torch.from_numpy(
                transitions["ee6D_t"][idx][:3]
            ).float()
            target_trajectory["end_effector_rotation"][i] = torch.from_numpy(
                transitions["ee6D_t"][idx][3:]
            ).float()
            target_trajectory["magnet"][i] = torch.from_numpy(
                np.atleast_1d(transitions["mgt_t"][idx]),
            ).float()
            target_trajectory["goal_obj6D_position"][i] = torch.from_numpy(
                transitions["goal_obj6D_t"][idx][:3]
            ).float()
            target_trajectory["goal_obj6D_rotation"][i] = torch.from_numpy(
                transitions["goal_obj6D_t"][idx][3:]
            ).float()
            target_trajectory["obstacle6D_position"][i] = torch.from_numpy(
                transitions["obstacle6D_t"][idx][:3]
            ).float()
            target_trajectory["obstacle6D_rotation"][i] = torch.from_numpy(
                transitions["obstacle6D_t"][idx][3:]
            ).float()

        return {
            "input": x,
            "initial_state_configuration": initial_state_configuration,
            "initial_state_end_effector": initial_state_end_effector,
            "initial_state_magnet": initial_state_mgt,
            "initial_state_goal_obj6D": initial_state_goal_obj6D,
            "initial_state_obstacle6D": initial_state_obstacle6D,
            "final_state_configuration": final_state_configuration,
            "final_state_end_effector": final_state_end_effector,
            "final_state_magnet": final_state_mgt,
            "final_state_goal_obj6D": final_state_goal_obj6D,
            "final_state_obstacle6D": final_state_obstacle6D,
            "trajectory_configuration": target_trajectory["configuration"],
            "trajectory_end_effector_position": target_trajectory["end_effector_position"],
            "trajectory_end_effector_rotation": target_trajectory["end_effector_rotation"],
            "trajectory_magnet": target_trajectory["magnet"],
            "trajectory_goal_obj6D_position": target_trajectory["goal_obj6D_position"],
            "trajectory_goal_obj6D_rotation": target_trajectory["goal_obj6D_rotation"],
            "trajectory_obstacle6D_position": target_trajectory["obstacle6D_position"],
            "trajectory_obstacle6D_rotation": target_trajectory["obstacle6D_rotation"],
        }

    def close(self):
        h5_file = getattr(self, "h5_file", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass
            finally:
                self.h5_file = None

    def __del__(self):
        self.close()

class ForwardModelDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        datapoint_indices: List[int],
    ):
        """
        Forward model training and validation dataset. Adjusted for parallel reading.

        :param h5_path: Path to h5 file with transition dataset.
        :param datapoint_indices: Indices of datapoints in transition dataset.
        """
        self.h5_path = h5_path
        self.datapoint_indices = datapoint_indices
        self.h5_file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.h5_file = None

    def _get_h5_file(self) -> h5py.File:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.h5_file

    def __len__(self) -> int:
        return len(self.datapoint_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        datapoint_idx = self.datapoint_indices[idx]

        h5_file = self._get_h5_file()
        transitions = h5_file["transitions"]

        x = torch.from_numpy(
            np.concatenate(
                (
                    transitions["joints_angles_t"][datapoint_idx],
                    transitions["ee6D_t"][datapoint_idx],
                    np.asarray([transitions["mgt_t"][datapoint_idx]]),
                    transitions["goal_obj6D_t"][datapoint_idx],
                    transitions["obstacle6D_t"][datapoint_idx],
                    transitions["desired_delta_q"][datapoint_idx],
                    np.asarray([transitions["delta_mgt"][datapoint_idx]]),
                )
            )
        ).float()


        next_configuration = torch.from_numpy(
            transitions["joints_angles_t1"][datapoint_idx]
        ).float()
        next_end_effector = torch.from_numpy(
            transitions["ee6D_t1"][datapoint_idx]
        ).float()
        next_magnet = torch.from_numpy(
            np.asarray([transitions["mgt_t1"][datapoint_idx]])
        ).float()
        next_goal_obj6D = torch.from_numpy(
            transitions["goal_obj6D_t1"][datapoint_idx]
        ).float()
        next_obstacle6D = torch.from_numpy(
            transitions["obstacle6D_t1"][datapoint_idx]
        ).float()

        return {
            "input": x,
            "next_configuration": next_configuration,
            "next_end_effector": next_end_effector,
            "next_magnet": next_magnet,
            "next_goal_obj6D": next_goal_obj6D,
            "next_obstacle6D": next_obstacle6D,
        }

    def close(self):
        h5_file = getattr(self, "h5_file", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass
            finally:
                self.h5_file = None

    def __del__(self):
        self.close()

class InverseModelDataset(Dataset):

    def __init__(
            self,
            h5_path: str,
            datapoint_indices: List[int],
    ):
        """
        Inverse model training and validation dataset. Adjusted for parallel reading.

        :param h5_path: Path to h5 file with transition dataset.
        :param datapoint_indices: Indices of datapoints in transition dataset.
        """

        self.h5_path = h5_path
        self.datapoint_indices = datapoint_indices
        self.h5_file = None

    def _get_h5_file(
            self
    ) -> h5py.File:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def __len__(
            self
    ) -> int:
        return len(self.datapoint_indices)

    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, torch.Tensor]:
        datapoint_idx = self.datapoint_indices[idx]

        h5_file = self._get_h5_file()
        transitions = h5_file["transitions"]

        x = torch.from_numpy(
            np.concatenate(
                (
                    transitions["joints_angles_t"][datapoint_idx],
                    transitions["ee6D_t"][datapoint_idx],
                    np.asarray([transitions["mgt_t"][datapoint_idx]]),
                    transitions["goal_obj6D_t"][datapoint_idx],
                    transitions["obstacle6D_t"][datapoint_idx],
                    transitions["ee6D_t1"][datapoint_idx],
                    transitions["goal_obj6D_t1"][datapoint_idx],
                    transitions["obstacle6D_t1"][datapoint_idx],
                )
            )
        ).float()

        action = torch.from_numpy(
            np.concatenate(
                (
                    transitions["desired_delta_q"][datapoint_idx],
                    np.asarray([transitions["delta_mgt"][datapoint_idx]]),
                )
            )
        ).float()

        return {
            "input": x,
            "action": action,
        }

    def __del__(
            self
    ):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
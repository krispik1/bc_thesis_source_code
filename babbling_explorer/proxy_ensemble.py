from typing import List

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from remake.dataset_types import Transition, Action, State

def flatten_state(
        state: State,
) -> np.ndarray:
    """
    Flattens state vector to be used as input for MLPs.

    :param state: State vector.
    :return: A flattened state vector as array.
    """
    joints_angles = state.joints_angles.ravel()
    end_effector6D = state.end_effector6D.ravel()
    goal_object6D = state.goal_object6D.ravel()
    obstacle6D = state.obstacle6D.ravel()
    magnet_state = np.array([float(state.magnet_state)], dtype=np.float64)

    return np.concatenate([joints_angles, end_effector6D, goal_object6D, obstacle6D, magnet_state], axis=0)

def flatten_action(
        action: Action,
) -> np.ndarray:
    """
    Flattens action vector to be used as input for MLPs.

    :param action: Action vector.
    :return: A flattened action vector as array.
    """
    delta_q = action.delta_q.ravel()
    delta_mgt = np.array([float(action.delta_mgt)], dtype=np.float64)

    return np.concatenate([delta_q, delta_mgt], axis=0)

class MLP(nn.Module):

    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            hidden_dimension: int = 256
    ):
        """
        Class representing simple multi-layer perceptron that is used in proxy ensemble.

        :param input_dimension: Dimension of the input.
        :param output_dimension: Dimension of the output.
        :param hidden_dimension: Dimension of the hidden layer.
        """
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(
            self,
            x
    ):
        return self.net(x)

class ProxyEnsemble:

    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            hidden_dimension: int = 256,
            n_models: int = 5,
            lr: float = 1e-3,
            device: str = 'cpu'
    ):
        """
        Proxy ensemble that is used in exploration of the environment. Their disagreement is used to measure uncertainty
        and provide epistemic motivation for moving through the environment. The models represent forward models that
        take state and action and predict next state.

        The ensemble consists of number of multi-layered perceptron models periodically trained during the exploration.

        :param input_dimension: Dimension of the input of MLPs.
        :param output_dimension: Dimension of the output of MLPs.
        :param hidden_dimension: Dimension of the hidden layer of MLPs.
        :param n_models: Number of MLPs.
        :param lr: Learning rate of MLPs.
        :param device: Device of MLPs.
        """

        self.device = device
        self.models = [MLP(input_dimension, output_dimension, hidden_dimension).to(device) for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.loss_function = nn.MSELoss()

    @torch.no_grad()
    def disagreement(
            self,
            state: State,
            action: Action
    ) -> float:
        """
        Used to get disagreement of the ensemble for a candidate action to take. The disagreement provides information
        about how uncertain the ensemble is about given state and action input. Higher the disagreement, higher the
        uncertainty and novelty of the state and action combination, which indicates unknown environment that should
        be explored.

        :param state: State vector.
        :param action: Action vector.
        :return: Disagreement metric provided by the ensemble.
        """

        # Flatten vectors to serve as input
        state = flatten_state(state)
        action = flatten_action(action)

        x = np.concatenate((state, action), axis=0).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device).unsqueeze(0)

        # Get predictions from each model
        predictions = []
        for model in self.models:
            predictions.append(model(xt).cpu().numpy().squeeze())
        predictions = np.stack(predictions, axis=0)

        # Return disagreement which is mean variance of predictions
        return float(np.mean(np.var(predictions, axis=0)))

    def train(
            self,
            batch: List[Transition],
            steps: int = 1
    ) -> None:
        """
        Used to train the models in the ensemble periodically. It is trained on batch of transitions provided by
        replay buffer.

        :param batch: Batch of transitions.
        :param steps: Number of training steps.
        """

        if len(batch) < steps:
            return

        for _ in range(steps):
            input_states = []
            output_states = []

            for transition in batch:
                state_t = flatten_state(transition.state_t)
                action = flatten_action(transition.action)
                state_t1 = flatten_state(transition.state_t1)
                input_states.append(np.concatenate([state_t, action], axis=0))
                output_states.append(state_t1)

            inputs = torch.from_numpy(np.stack(input_states, axis=0).astype(np.float32)).to(self.device)
            outputs = torch.from_numpy(np.stack(output_states, axis=0).astype(np.float32)).to(self.device)

            for model, optimizer in zip(self.models, self.optimizers):
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = self.loss_function(predictions, outputs)
                loss.backward()
                optimizer.step()
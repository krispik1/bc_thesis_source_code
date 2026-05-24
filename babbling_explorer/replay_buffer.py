import random
from typing import List

from remake.dataset_types import Transition

class ReplayBuffer:

    def __init__(
            self,
            capacity: int = 200000
    ):
        """
        Class representing a replay buffer where we store transitions that are later sampled and used to train MLPs
        in the proxy ensemble. The buffer behaves as circle buffer, meaning if it reaches full capacity, it starts
        replacing transitions by moving the index/pointer to start of list.

        :param capacity: Capacity of the replay buffer.
        """

        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.index = 0

    def add(
            self,
            transition: Transition
    ) -> None:
        """
        Adds a transition to the replay buffer.

        :param transition: Transition object.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Circle back to beginning
            self.buffer[self.index] = transition
            self.index = (self.index + 1) % self.capacity

    def sample(
            self,
            batch_size: int
    ) -> List[Transition]:
        """
        Used to get batch of transitions to be used for training the proxy ensemble.

        :param batch_size: Number of transitions to be sampled.
        :return: List of sampled transitions.
        """
        return random.sample(
            population=self.buffer,
            k=min(batch_size, len(self.buffer))
        )

    def __len__(self) -> int:
        return len(self.buffer)
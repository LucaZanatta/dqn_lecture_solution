import torch
import numpy as np
from .memory import ReplyMemory
from .exploration import EpsilonGreedy
from .rl_algo import LearnDQN


class AgentOffPolicy:
    """Agent that takes care of the whole setup."""

    def __init__(self, net: torch.nn.Module, memory: ReplyMemory, exploration: EpsilonGreedy, learn_algo: LearnDQN, iterations_number: int):
        """
        Creates the class attributs.

        Args:
            net: network to be trained.
            memory: reply memory where the trajectories are saved.
            exploration: exploration class that takes care of the exploratio-exploitation dilemma.
            learn_algo: learning algorithm. It takes care of the learning function.
            iterations_number: number of "mini epochs" in the training.
        """
        self.net = net
        self.memory = memory
        self.exploration = exploration
        self.learn_algo = learn_algo
        self.iterations_number = iterations_number

    def choose_action(self, state: np.array) -> int:
        """
        Chooses the action.

        Args:
            state: the state of the environment.
        
        Returns:
            action: the choosen action.
        """
        action = self.exploration.random_action()
        if action is None:
            with torch.no_grad():
                action = torch.argmax(self.net(torch.tensor(state))).item()
        return action

    def learn(self) -> None:
        """Takes care of the learning of the network."""
        if self.memory.idx_memory >= self.memory.batch_sz:
            states, actions, rewards, states_, dones = self.memory.batch()
            for _ in range(self.iterations_number):
                for i in range(self.memory.batch_numbers):
                    y, y_hat = self.learn_algo(self.net, states[i], actions[i], rewards[i], states_[i], dones[i])
                    self.net.learn(y, y_hat)
       
    def memory_add(self, state: np.array, action: int, reward: float, state_: np.array, done: int) -> None:
        """
        Add a new sample into memory.

        Args:
            state: the current state of the environment.
            action: the action taken.
            reward: the reward collected from that action.
            state_: the future state.
            done: if the game is terminated.
        """
        self.memory.add(state, action, reward, state_, done)

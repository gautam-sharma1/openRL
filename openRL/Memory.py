###########################################################
# Memory.py                                               #
# Defines templated classes to store RL data             #
# @author Gautam Sharma                                   #
###########################################################

__author__ = "Gautam Sharma"


from collections import deque
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class Memory(ABC):
    def __init__(self):
        super().__init__()
        batch_size: int
        max_size: int

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def get_random(self):
        pass


class ReplayBuffer(Memory):
    def __init__(self, batch_size=32, size=100000):
        super().__init__()
        self.memory = deque(maxlen=size)  # popleft()
        self.max_size = size
        self.batch_size = batch_size
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_random(self):
        return random.sample(self.memory, self.batch_size)


class ReplayBufferImages(ReplayBuffer):
    def __init__(self, transform, batch_size=32, size=100000):
        super().__init__(batch_size, size)
        self.preprocess = transform

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

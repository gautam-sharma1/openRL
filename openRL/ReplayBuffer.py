from collections import deque
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Memory(ABC):
    def __init__(self):
        super().__init__()
        batch_size : int
        max_size : int
    
    @abstractmethod
    def add(self,state, action, reward, next_state, done):
        pass

    @abstractmethod
    def get_random(self):
        pass


class ReplayBuffer(Memory):
    def __init__(self, batch_size = 32, size= 100000):
        super().__init__()
        self.memory = deque(maxlen=size) # popleft()
        self.max_size = size
        self.batch_size= batch_size
        self.pos = 0
    
    # TODO
    def add(self,state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
          
    def get_random(self):
        return zip(*(random.sample(self.memory, self.batch_size)))
    
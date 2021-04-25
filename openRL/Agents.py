###########################################################
# Agents.py                                               #
# Defines templated class that needs to be inherited from #
# @author Gautam Sharma                                   #
###########################################################

__author__ = "Gautam Sharma"


from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Template for defining a Reinforcement Learning agent
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def terminate(self, *args, **kwargs):
        pass

    @abstractmethod
    def graphics(self, *args, **kwargs):
        pass

    @abstractmethod
    def take_action(self, *args, **kwargs):
        pass

    @staticmethod
    def log_data(*args, **kwargs):
        pass

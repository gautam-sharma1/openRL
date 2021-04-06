# import torchvision.transforms as T
# from PIL import Image
from openRL.AI import Net, DQL, DQN
from openRL.ReplayBuffer import ReplayBuffer
import pygame
import torch
import math
import random
import torch.autograd as autograd
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

directions = {"RIGHT": 0, "LEFT": 1, "UP": 2, "DOWN": 3}
dir_list = ["RIGHT", "LEFT", "UP", "DOWN"]

SPEED = 10


class Agent(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def terminate(self):
        pass

    @abstractmethod
    def graphics(self):
        pass

    @abstractmethod
    def take_action(self, action):
        pass

    @staticmethod
    def log_data(input):
        pass



class Robot(Agent):
    def __init__(self,
                 width,
                 height,
                 robot_color=(0, 0, 255),
                 goal_color=(200, 0, 0), obstacle_color=(0, 255, 0)):
        super().__init__()
        self.width = width
        self.height = height
        self.robot_color = robot_color
        self.goal_color = goal_color
        self.obstacle_color = obstacle_color
        self.display = pygame.display.set_mode((self.width, self.height))
        self.step_size = 40
        self.reset()
        self.clock = pygame.time.Clock()
        self.reward = 0

    __directions = {"RIGHT": 0, "LEFT": 1, "UP": 2, "DOWN": 3}
    __dir_list = ["RIGHT", "LEFT", "UP", "DOWN"]

    def get_state(self):
        # x,y,dx,dy
        return torch.Tensor([self.robot_pos[0], self.robot_pos[1], self.goal_pos[0]-self.robot_pos[0], self.goal_pos[1]-self.robot_pos[1]])

    def __get_random_pos(self):
        return [
            random.randint(0, self.width - self.step_size),
            random.randint(0, self.height - self.step_size)
        ]

    def reset(self):
        self.robot_pos = self.__get_random_pos()  # TODO
        self.dir = 0  # right
        self.reward = 0
        self.goal_pos = self.__get_random_pos()
        while self.robot_pos[0] == self.goal_pos[0] or self.robot_pos[
                1] == self.goal_pos[1]:
            self.goal_pos = self.__get_random_pos()
        return self.get_state()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move
        self.take_action(action)  # goes to new position
        self.reward = 0
        done = False
        reward = 0

        # went outside the boundary
        if self.terminate():
            self.reward = -10
            # print("Terminating!")
            done = True
            # self.reset()  # TODO: only for testing
            self.graphics()
            self.clock.tick(SPEED)
            return self.get_state(), reward, done

        # reached goal
        elif abs(self.robot_pos[0] -
                 self.goal_pos[0]) <= self.step_size and abs(
                     self.robot_pos[1] - self.goal_pos[1]) <= self.step_size:
            reward = 10
            done = True
            self.graphics()
            self.clock.tick(SPEED)
            return self.get_state(), reward, done

        else:
            self.graphics()
            self.clock.tick(SPEED)

            #           s' , r, done
            return self.get_state(), -0.1, done

    def terminate(self):
        if self.robot_pos[0] < 0 or self.robot_pos[0] > self.width-self.step_size/2 or \
                self.robot_pos[1] < 0 or self.robot_pos[1] > self.height-self.step_size/2:
            return True
# TODO:
        return False

    def draw_players(self, pos, color):
        pygame.draw.rect(
            self.display, color,
            pygame.Rect(pos[0], pos[1], self.step_size, self.step_size))

    def graphics(self):
        self.display.fill((0, 0, 0))
        self.draw_players(self.robot_pos, self.robot_color)  # draw robot
        self.draw_players(self.goal_pos, self.goal_color)  # draw goal
        pygame.display.flip()

    # TODO
    def take_action(self, action):

        if action == 0:  # RIGHT
            self.robot_pos[0] += self.step_size
        elif action == 1:  # LEFT
            self.robot_pos[0] -= self.step_size

        elif action == 2:  # UP
            self.robot_pos[1] -= self.step_size
        elif action == 3:  # DOWN
            self.robot_pos[1] += self.step_size


# note: Run any one of the below algos. Be sure to comment the other one out
###################################
# Q-Learning with neural network  #
###################################
# remember = ReplayBuffer(64, 100000)
# r = Robot(400, 400)
# model = DQL(0.99, 4, 4)
# losses = []
# NUM_EPISODES = 100000
# episode_reward = 0
#
# state = r.get_state()
# all_rewards = []
# alpha = 0.99
#
# for j in range(1, NUM_EPISODES):
#
#     action = model.act(state)
#     next_state, reward, done = r.step(action)
#     remember.add(state, action, reward, next_state, done)
#     state = r.get_state()  # s'
#     episode_reward += reward
#
#     if done:
#         state = r.reset()
#         all_rewards.append(episode_reward)
#         episode_reward = 0
#
#     if len(remember.memory) > remember.batch_size:
#         loss = model.compute_loss(remember)
#         losses.append(loss.item())
#
#     if j % 100 == 0:
#         print(j, model.epsilon, sum(all_rewards) /
#               len(all_rewards), sum(losses)/len(losses))
#
#         model.epsilon *= alpha


# ###########################
# # DQN
# ##########################
PATH = './net.pth'
remember = ReplayBuffer(100, 10000)
r = Robot(600, 600)
model = DQN(0.99, 4, 4)
losses = []
NUM_EPISODES = 1000000
episode_reward = 0


all_rewards = []
max_reward = 0
alpha = 0.99
NUM_TRAJECTORY = 20

for j in range(1, NUM_EPISODES):

    state = r.reset()
    for t in range(1,NUM_TRAJECTORY):

        action = model.act(state)
        next_state, reward, done = r.step(action)
        remember.add(state, action, reward, next_state, done)
        state = r.get_state()  # s'
        episode_reward += reward

        if done:
            state = r.reset()

            if episode_reward > max_reward:
                max_reward = episode_reward
                model.save_model(PATH)
            all_rewards.append(episode_reward)
            episode_reward = 0


        if len(remember.memory) > remember.batch_size:
            loss = model.compute_loss(remember)
            losses.append(loss.item())

    if j % 200 == 0:
        model.epsilon *= alpha
        print(j, model.epsilon, sum(all_rewards)/len(all_rewards), sum(losses)/len(losses))
        model.copy_weights()

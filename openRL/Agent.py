# import torchvision.transforms as T
# from PIL import Image
from openRL.AI import Net, DQL, DQN
from openRL.ReplayBuffer import ReplayBuffer
import pygame
import torch
import math
import random
from collections import namedtuple
from enum import Enum
import torch.autograd as autograd
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

directions = {"RIGHT": 0, "LEFT": 1, "UP": 2, "DOWN": 3}
dir_list = ["RIGHT", "LEFT", "UP", "DOWN"]
Point = namedtuple('Point', 'x, y')

SPEED = 40

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

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
        self.step_size = 20
        self.side_size = 20
        self.reset()
        self.clock = pygame.time.Clock()
        self.reward = 0
        self.n_games = 0

    __directions = {"RIGHT": 0, "LEFT": 1, "UP": 2, "DOWN": 3}
    __dir_list = ["RIGHT", "LEFT", "UP", "DOWN"]

    # TODO: Need to change this since it's copied.
    def get_state(self):

        head = self.robot_pos
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.terminate(point_r)) or
            (dir_l and self.terminate(point_l)) or
            (dir_u and self.terminate(point_u)) or
            (dir_d and self.terminate(point_d)),

            # Danger right
            (dir_u and self.terminate(point_r)) or
            (dir_d and self.terminate(point_l)) or
            (dir_l and self.terminate(point_u)) or
            (dir_r and self.terminate(point_d)),

            # Danger left
            (dir_d and self.terminate(point_r)) or
            (dir_u and self.terminate(point_l)) or
            (dir_r and self.terminate(point_u)) or
            (dir_l and self.terminate(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location

            self.goal_pos.x < self.robot_pos.x,  # food left
            self.goal_pos.x > self.robot_pos.x,  # food right
            self.goal_pos.y < self.robot_pos.y,  # food up
            self.goal_pos.y > self.robot_pos.y  # food down
        ]
        return torch.Tensor(state)

    # def get_state(self):
    #     # x,y,dx,dy
    #     return torch.Tensor([self.robot_pos.x, self.robot_pos.y, self.robot_pos.x > self.goal_pos.x, self.robot_pos.y>self.goal_pos.y, self.goal_pos.x - self.robot_pos.x,
    #                          self.goal_pos.y - self.robot_pos.y])

    def __get_random_pos(self):
        return Point(
            random.randint(0, (self.width - self.side_size) // self.side_size)*self.side_size,
            random.randint(0, (self.height - self.side_size) // self.side_size)*self.side_size
        )

    def reset(self):
        self.robot_pos = Point(100,100) # TODO
        self.direction = Direction.RIGHT  # right
        self.reward = 0
        self.goal_pos = self.__get_random_pos()
        # while self.robot_pos.x == self.goal_pos[0] and self.robot_pos[
        #         1] == self.goal_pos[1]:
        #     self.goal_pos = self.__get_random_pos()
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
            reward = -10
            # print("Terminating!")
            done = True
            # self.reset()  # TODO: only for testing
            # self.graphics()
            # self.clock.tick(SPEED)
            return self.get_state(), reward, done

        # reached goal
        elif self.robot_pos == self.goal_pos:
            reward = 10
            #reward = 1/(abs(self.robot_pos[0] - self.goal_pos[0]) +abs(self.robot_pos[1] - self.goal_pos[1]) + 0.01 )
            # self.graphics()
            #self.clock.tick(SPEED)
            self.move_goal()
            return self.get_state(), reward, done

        else:
            #reward = 1 / (abs(self.robot_pos[0] - self.goal_pos[0]) + abs(self.robot_pos[1] - self.goal_pos[1]) + 0.01)
            self.graphics()
            self.clock.tick(SPEED)

            #           s' , r, done
            return self.get_state(), 0, done

    # TODO: Need to change this since it's copied
    def terminate(self,pt=None):
        # if self.robot_pos[0]-self.side_size/2 < 0 or self.robot_pos[0] > self.width-self.side_size/2 or \
        #         self.robot_pos[1]-self.side_size/2  < 0 or self.robot_pos[1] > self.height-self.side_size/2:
        #     return True
        if pt is None:
            pt = self.robot_pos
        if pt.x > self.width - self.side_size or pt.x < 0 or pt.y > self.height - self.side_size or pt.y < 0:
            return True

        return False

    # This is fine
    def draw_players(self, pos, color):
        pygame.draw.rect(
            self.display, color,
            pygame.Rect(pos.x, pos.y, self.side_size, self.side_size))
    # This is fine
    def graphics(self):
        self.display.fill((0, 0, 0))
        self.draw_players(self.robot_pos, self.robot_color)  # draw robot
        self.draw_players(self.goal_pos, self.goal_color)  # draw goal
        pygame.display.flip()

    def move_goal(self):
        self.goal_pos = self.__get_random_pos()

    # TODO Need to change this as it is copied from online source
    def take_action(self, action):
            # [straight, right, left]

            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

            self.direction = new_dir

            x = self.robot_pos.x
            y = self.robot_pos.y
            if self.direction == Direction.RIGHT:
                x += self.side_size
            elif self.direction == Direction.LEFT:
                x -= self.side_size
            elif self.direction == Direction.DOWN:
                y += self.side_size
            elif self.direction == Direction.UP:
                y -= self.side_size

            self.robot_pos = Point(x, y)
    # def take_action(self, action):
    #     x = self.robot_pos.x
    #     y = self.robot_pos.y
    #     if action == 0:  # RIGHT
    #         x += self.step_size
    #     elif action == 1:  # LEFT
    #         x -= self.step_size
    #
    #     elif action == 2:  # UP
    #         y -= self.step_size
    #     elif action == 3:  # DOWN
    #         y += self.step_size
    #
    #     # elif action == 4:  # DIAGONAL LOWER RIGHT
    #     #     self.robot_pos[0] += self.step_size
    #     #     self.robot_pos[1] += self.step_size
    #     #
    #     # elif action == 5:  # DIAGONAL LOWER LEFT
    #     #     self.robot_pos[0] -= self.step_size
    #     #     self.robot_pos[1] += self.step_size
    #     #
    #     # elif action == 6:  # DIAGONAL UPPER LEFT
    #     #     self.robot_pos[0] -= self.step_size
    #     #     self.robot_pos[1] -= self.step_size
    #     #
    #     # elif action == 7:  # DIAGONAL UUPER RUGHT
    #     #     self.robot_pos[0] += self.step_size
    #     #     self.robot_pos[1] -= self.step_size
    #     #
    #     # else:
    #     #     self.robot_pos[0] += 0
    #     #     self.robot_pos[1] += 0
    #
    #     self.robot_pos = Point(x, y)


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
PATH = './net'
remember = ReplayBuffer(10, 1000000)
r = Robot(400, 400)

model = DQL(1, 11, 3)
losses = []
NUM_EPISODES = 20000
episode_reward = 0


all_rewards = []
max_reward = 0
alpha = 0.99
state = r.reset()
print(state)

NUM_TRAJECTORY = 20

while True:

    state = r.get_state()
    action = model.act(state)
    next_state, reward, done = r.step(action)
    model.compute_single_loss(state, action, reward, next_state, done)
    #model.train_short_memory(state, action, reward, next_state, done)
    remember.add(state, action, reward, next_state, done)
    #state = r.get_state()  # s'

    #episode_reward += reward
    if reward > max_reward:
        max_reward = reward
        model.save_model(PATH)
        # print(j, model.epsilon, reward)
        print('saving!...')
    all_rewards.append(reward)

    if done :
        state = r.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        model.n_games += 1


        #model.train_long_memory(remember)
        # loss = model.compute_loss(remember)
        # losses.append(loss.item())
    if len(remember.memory) > remember.batch_size:
        state, action, reward, next_state, done = zip(*remember.get_random())
        loss = model.compute_loss(state, action, reward, next_state, done)
        losses.append(loss.item())



plt.plot(all_rewards)
plt.plot(losses)
########
# Eval script
########
# NUM_EPISODES = 100
# r = Robot(200, 200)
# model = DQL(0, 6, 4)
# # state = r.reset()
# model.model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
# model.model.eval()
# prev_action = -1
# state = r.reset()
# with torch.no_grad():
#     for j in range(1, NUM_EPISODES):
#
#
#         avg_reward = 0
#         move = False
#         state = r.reset()
#         for i in range(1,100):
#
#             action = model.model(state)
#             print(action)
#
#             action = torch.argmax(action)
#             #
#             # if prev_action == 0 and action == 1 or action ==1 and prev_action == 0 or\
#             #     prev_action == 2 and action == 3 or action ==3 and prev_action == 2or action == 5:
#             #     new_action = action
#             #     while new_action == action:
#             #         new_action = random.randint(0, 3)
#             #     action = new_action
#
#             prev_action = action
#             next_state, reward, done = r.step(action)
#             print(reward)
#             #
#             if i>50 and not move:
#                 r.move_goal()
#                 move = True
#             avg_reward += reward
#
#             if reward == 10:
#                 break
#
#             if done:
#                 break
#             state = r.get_state()



    # #if j%50==0:
    # model.epsilon *= alpha
    # if j % 100 == 0:
    #     # model.copy_weights()
    #     #if j > 512:
    #     print(j, model.epsilon, episode_reward, sum(losses)/len(losses))
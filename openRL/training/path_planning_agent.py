import random
from collections import namedtuple
from enum import Enum

import pygame
import torch
import torchvision.transforms as T
from PIL import Image

from openRL.AI import DQL
from openRL.Agents import Agent
from openRL.Memory import ReplayBuffer

directions = {"RIGHT": 0, "LEFT": 1, "UP": 2, "DOWN": 3}
dir_list = ["RIGHT", "LEFT", "UP", "DOWN"]
Point = namedtuple('Point', 'x, y')

SPEED = 40


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Robot(Agent):
    def __init__(self,
                 width,
                 height,
                 robot_color=(0, 0, 255),
                 obstacle_color=(200, 0, 0), goal_color=(0, 255, 0)):
        super().__init__()
        self.width = width
        self.height = height
        self.robot_color = robot_color
        self.goal_color = goal_color
        self.obstacle_color = obstacle_color
        self.obstacle_list = []
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Path Planning using RL")
        self.step_size = 20
        self.side_size = 20
        self.num_obstacles = 10
        self.preprocess = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(40, interpolation=Image.CUBIC),
            T.ToTensor()])
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

        extra_state = [
            [pos.x < self.robot_pos.x, pos.x > self.robot_pos.x, pos.y < self.robot_pos.y, pos.y > self.robot_pos.y] for
            pos in self.obstacle_list]

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

            # Goal location
            self.goal_pos.x < self.robot_pos.x,  # food left
            self.goal_pos.x > self.robot_pos.x,  # food right
            self.goal_pos.y < self.robot_pos.y,  # food up
            self.goal_pos.y > self.robot_pos.y,  # food down

        ]

        for entries in extra_state:
            state.extend(entries)

        return torch.Tensor(state)


    def __get_random_pos(self):
        return Point(
            random.randint(0, (self.width - self.side_size) // self.side_size) * self.side_size,
            random.randint(0, (self.height - self.side_size) // self.side_size) * self.side_size
        )

    def reset(self):
        self.robot_pos = Point(100, 100)  # TODO
        self.direction = Direction.RIGHT  # right
        self.reward = 0
        self.goal_pos = self.__get_random_pos()
        self.obstacle_list = [self.__get_random_pos() for _ in range(self.num_obstacles)]

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
        if self.goal_pos in self.obstacle_list:
            self.reset()
            pass

        if self.terminate():
            reward = -10
            done = True
            return self.get_state(), reward, done

        elif self.robot_pos == self.goal_pos:
            reward = 10
            self.move_goal()
            return self.get_state(), reward, done

        else:
            self.graphics()
            self.clock.tick(SPEED)

            #           s' , r, done
            return self.get_state(), 0, done

    # TODO: Need to change this since it's copied
    def terminate(self, pt=None):
        if pt is None:
            pt = self.robot_pos
        if pt.x > self.width - self.side_size or pt.x < 0 or pt.y > self.height - self.side_size or pt.y < 0:
            return True
        elif pt in self.obstacle_list:
            return True

        return False

    # This is fine
    def draw_players(self, pos, color):
        pygame.draw.rect(
            self.display, color,
            pygame.Rect(pos.x, pos.y, self.side_size, self.side_size))
        # self.display.blit(image, (pos.x - 16, pos.y - 16))

    # This is fine
    def graphics(self):
        self.display.fill((0, 0, 0))
        self.draw_players(self.robot_pos, self.robot_color)  # draw robot
        self.draw_players(self.goal_pos, self.goal_color)  # draw goal
        for pt in self.obstacle_list:
            self.draw_players(pt, self.obstacle_color)  # draw goal
        pygame.display.flip()

    def move_goal(self):
        self.goal_pos = self.__get_random_pos()

    def take_action(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 1:
            new_dir = clock_wise[idx]  # no change
        elif action == 2:
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


# ###########################
# # DQL
# ##########################
if __name__ == "__main__":
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((40, 40), interpolation=Image.CUBIC),
        T.ToTensor()])

    PATH = './net'
    remember = ReplayBuffer(100, 1000000)
    r = Robot(400, 400)

    model = DQL(1, 51, 3)

    losses = []
    NUM_EPISODES = 20000
    episode_reward = 0

    all_rewards = []
    max_reward = 0
    alpha = 0.99
    # state = r.reset()

    NUM_TRAJECTORY = 20
    j = 0
    while True:

        j += 1
        state = r.reset()
        for i in range(40):
            state = r.get_state()  # s'
            action = model.act(state)
            next_state, reward, done = r.step(action)
            model.compute_single_loss(state, action, reward, next_state, done)
            remember.add(state, action, reward, next_state, done)

            if reward > max_reward:
                max_reward = reward
                model.save_model(PATH)
                # print(j, model.epsilon, reward)
                print('saving!...')
            all_rewards.append(reward)

            if done:
                state = r.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                model.n_games += 1

            if len(remember.memory) > remember.batch_size:
                state, action, reward, next_state, done = zip(*remember.get_random())

                loss = model.compute_batch_loss(state, action, reward, next_state, done)

                losses.append(loss.item())

        if j % 200 == 0:
            model.epsilon *= alpha

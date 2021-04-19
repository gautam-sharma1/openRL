import pickle
from openRL.AI import Net32, DQL, DQN

from openRL.Agent import Agent
from openRL.ReplayBuffer import ReplayBuffer,ReplayBufferImages
import pygame
import pickle
import torch
import math
from openRL.pendulum import Environment
SPEED = 100
import random

if __name__ == "__main__":
    r = Environment()
    input_list = []
    output_list = []

    losses = []
    NUM_EPISODES = 20000
    episode_reward = 0


    all_rewards = []
    max_reward = 0
    alpha = 0.99
    state = r.reset()

    for i in range(10000):
        theta1 = random.randint(0,180)
        theta2 = random.randint(0,360)
        r.transform(theta1,theta2)
        input_list.append([math.radians(theta1), math.radians(theta2)])
        output_list.append([r.x2-200,r.y2-200])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        r.graphics()
        r.clock.tick(SPEED)


    assert len(input_list) == 10000
    with open('input.pkl', 'wb') as f:
        pickle.dump(input_list, f)

    assert len(output_list) == 10000
    with open('output.pkl', 'wb') as f:
        pickle.dump(output_list, f)
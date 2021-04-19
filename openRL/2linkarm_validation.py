import pickle
from openRL.AI import Net32,Net, DQL, DQN

from openRL.Agent import Agent
from openRL.ReplayBuffer import ReplayBuffer,ReplayBufferImages
import pygame
import torch
import math
from openRL.pendulum_training import Environment
SPEED = 100
import random


if __name__ == "__main__":
    r = Environment()
    net = Net32(2, 2)
    PATH = './models/net.pthnewest862'
    with open('input.pkl', 'rb') as f:
        output_list = pickle.load(f)

    with open('output.pkl', 'rb') as f:
        input_list = pickle.load(f)
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    with torch.no_grad():
        NUM_TRAJECTORY = 20
        j = 0
        for i in range(200, 250):
            x = input_list[i]
            r.goal_pos[0], r.goal_pos[1] = x[0] + 200, x[1] + 200
            print(input_list[i])
            output = net(torch.FloatTensor(x))
            print(f"Actual : {output}, Expected: {output_list[i]}")
            r.transform(math.degrees(output[0]), math.degrees(output[1]))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            r.graphics()
            r.clock.tick(SPEED)
            pygame.time.wait(2000)
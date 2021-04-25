from openRL.extras.pendulum_training import Environment
from openRL.AI import DQL
import torch
import pygame

def evaluation(PATH):
    r = Environment()
    model = DQL(1,4,2)
    model.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.model.eval()
    with torch.no_grad():
        j = 0
        while True:
            j += 1
            state = r.get_state()
            action = model.model(state)
            print(action)
            action = torch.argmax(action)
            next_state, reward, done = r.step(action)
            pygame.time.wait(100)

            if done:
                r.reset()

if __name__ == "__main__":
    PATH =  './models/pendulum.pth100'
    evaluation(PATH)


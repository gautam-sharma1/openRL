from openRL.pendulum_training import Environment
from openRL.ReplayBuffer import ReplayBuffer
from openRL.AI import DQL
import torch

def evaluation(PATH):
    r = Environment()
    model = DQL(1, 13,2)
    model.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.model.eval()
    state = r.reset()
    with torch.no_grad():
        j = 0
        while True:
            j += 1
            state = r.get_state()
            action = model.model(state)
            print(action)
            action = torch.argmax(action)
            next_state, reward, done = r.step(action)
            if j % 40 == 0:
                r.reset()
            if done:
                break

if __name__ == "__main__":

    PATH =  './models/pendulum.pth10'
    evaluation(PATH)
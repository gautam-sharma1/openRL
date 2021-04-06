import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import random
from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self,state):
        pass

    @abstractmethod
    def compute_loss(self, replay_buffer):
        pass
    
    @staticmethod
    def tuple_to_tensor(input):
        l = []
        for i in input:
            l.append([i])
        return torch.LongTensor(l)

    @abstractmethod
    def save_model(self,PATH):
        pass


# TODO:
class DDPG:
    pass

# TODO:
class PPO:
    pass


class DQN(Algorithm):
    def __init__(self, epsilon, input_dim, output_dim, optimizer='Adam', loss_fcn=nn.MSELoss):
        super().__init__()
        self.epsilon = epsilon
        self.optim = optim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actor = Net(input_dim, output_dim)
        self.critic = Net(input_dim, output_dim)
        self.critic.load_state_dict(self.actor.state_dict())
        self.optimizer = optim.Adam(self.actor.parameters())
        self.gamma = 0.8
        (self.actor.parameters)

    def act(self, state):
        if random.random() > self.epsilon:
            output = self.actor(state)
            return torch.argmax(output)
        else:
            return random.randint(0, self.output_dim-1)

    
    def copy_weights(self):
        self.critic.load_state_dict(self.actor.state_dict())



    def compute_loss(self, replay_buffer):
        
        state, action, reward, next_state, done = replay_buffer.get_random()
        state = torch.vstack(state)
        next_state = torch.vstack(next_state)
        t_action = self.tuple_to_tensor(action)
        t_reward = self.tuple_to_tensor(reward)
        t_done = self.tuple_to_tensor(done)
        q_values = self.actor(state)             # q(s)
        
        next_q_values = self.critic(next_state) # q(s')
       
        q_value = q_values.gather(1, t_action)
        next_q_value = next_q_values.max(1)[0]
        
        expected_q_value = t_reward + self.gamma * next_q_value * (torch.LongTensor([1]) - t_done)
        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss

    def save_model(self,PATH):
        torch.save(self.actor.state_dict(),PATH)

# usage model = DQL()

class DQL(Algorithm):                                                # width, height, output
    def __init__(self, epsilon, input_dim, output_dim, optimizer='Adam', loss_fcn=nn.MSELoss):
        super().__init__()
        self.epsilon = epsilon
        self.optim = optim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = Net(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.gamma = 0.8

    def act(self, state):
        if random.random() > self.epsilon:
            output = self.model(state)
            return torch.argmax(output)
        else:
            return random.randint(0, self.output_dim-1)


    def compute_loss(self, replay_buffer):
        
        state, action, reward, next_state, done = replay_buffer.get_random()
        state = torch.vstack(state)
        next_state = torch.vstack(next_state)
        t_action = self.tuple_to_tensor(action)
        t_reward = self.tuple_to_tensor(reward)
        t_done = self.tuple_to_tensor(done)
        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(1, t_action)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = t_reward + self.gamma * next_q_value * (torch.LongTensor([1]) - t_done)
        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss


class Net(nn.Module):  # TODO improve the class definition
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        #x = x.view(1, 2)
        x = F.leaky_relu(self.fc1(state))
        return self.fc2(x)

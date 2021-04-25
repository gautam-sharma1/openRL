"""
    pendulum_training.py
    --------------------

    This script performs the training of a simple pendulum

"""

from openRL.AI import DQL,DQN
from openRL.Memory import ReplayBuffer
from openRL.training.path_planning_agent import Agent
import pygame
import torch
import math
import random
import matplotlib.pyplot as plt
from IPython import display

plt.ion()


SPEED = 100

class Environment(Agent):
    """
    Read the documentation for the :meth:'circle' method
    """
    def __init__(self,robot_color=(0, 255, 0),l=70,w=10):
        """

        :param robot_color:
        :param l:
        :param w:
        """
        super().__init__()
        self.length_link1 = 100
        self.length_link2 = 35
        self.width = w
        self.robot_color = robot_color
        self.base_color = (255, 0, 0)
        self.goal_color = (255,50,255)
        self.display = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Simple Pendulum")
        self.step_size = 10
        self.goal_pos = [250, 250]

        self.clock = pygame.time.Clock()
        self.reward = 0
        self.n_games = 0
        self.theta1 = 0
        self.theta2 = 0
        self.center_link1 = (200,200)
        self.center_link2 = (200+ self.length_link1, 200)
        self.x1, self.y1, self.x2, self.y2 = 200,200,200+self.length_link2,200
        self.rotation = 0
        self.rotation2 = 0
        self.delta_x = 0
        self.delta_y = 0
        self.reset()

    def __get_random_pos_goal(self):
            return random.randint(self.center_link1[0]+self.length_link1-self.length_link2, self.center_link1[0]+self.length_link1)

    def inverse_kinematics(self):
        theta1 = self.theta1
        theta2 = self.theta2 - self.theta1
        a = ((self.goal_pos[0]-200)**2 + (self.goal_pos[1]-200)**2)
        c2 = (a -(self.length_link1**2 + self.length_link2**2))/(2*self.length_link1*self.length_link2)
        s2 = math.sqrt(1-c2**2)
        q2 = math.atan2(s2,c2)
        q1 = -math.atan2(self.goal_pos[1]-200, self.goal_pos[0]-200) - math.atan2(self.length_link2*s2,self.length_link1+self.length_link2*c2)
        return math.degrees(q1),math.degrees(q2)

    def circle(self):
        x = random.randint(self.center_link1[0]-self.length_link1, self.center_link1[0]+self.length_link1)
        y = math.sqrt((self.length_link1)**2 - (x-200)**2)
        y_choice = 200-y
        # choice = random.randint(0,1)
        return x,y_choice

    def terminate(self):
        pass

    # TODO
    def reset(self):
        self.goal_pos = list(self.circle())
        self.center_link1 = (200,200)
        self.center_link2 = (200 + self.length_link1, 200)
        self.x1, self.y1, self.x2, self.y2 = 200 + self.length_link1, 200, 200 + self.length_link1+self.length_link2, 200
        #self.theta1 = 0
        return self.get_state()

    def transform(self,theta1):
        """

        :param theta1:
        :param theta2:
        :return: The new value of :attr:`x1` and :attr:`y1`
        """
        self.x1 = self.center_link1[0] + self.length_link1*math.cos(math.radians(theta1))
        self.y1 = self.center_link1[1] + self.length_link1*math.sin(math.radians(-theta1))

    def graphics(self):
        self.display.fill((0, 0, 0))
        self.draw_players()
        pygame.display.flip()

    def draw_players(self):
        pygame.draw.line(self.display,self.robot_color,self.center_link1,(self.x1,self.y1),10)
        pygame.draw.circle(self.display,self.base_color,self.center_link1,2,1)
        pygame.draw.circle(self.display, self.base_color, (int(self.x1),int(self.y1)), 2, 1)
        pygame.draw.circle(self.display, self.goal_color, (int(self.goal_pos[0]), int(self.goal_pos[1])), 6, 2)

    def get_state(self):
            state = [
                     self.theta1,
                     self.rotation,
                     #self.x1,
                     #self.y1,
                     # (self.goal_pos[0]-self.x1),
                     # (self.goal_pos[1]-self.y1),
                     (self.goal_pos[0]),
                     (self.goal_pos[1]),
                     # self.x1 > self.goal_pos[0],
                     # self.x1 < self.goal_pos[0],
                     # self.y1 > self.goal_pos[1],
                     # self.y1 < self.goal_pos[1],
            ]

            return torch.FloatTensor(state)


    def move_goal(self):
        self.goal_pos = self.__get_random_pos_goal()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        self.take_action(action)  # goes to new position
        self.reward = 0
        done = False
        reward = 0
        if self.theta1 > 180:
            self.theta1 = 180
            #return self.get_state(), -10, True
            self.transform(self.theta1)


        if self.theta1 < 0:
            self.theta1 = 0
            #return self.get_state(), -10,True
            #self.transform(self.theta1)


        if self.goal_reached():
                    print("Goal reached!")
                    reward = 10
                    done = True
                    state = self.reset()
                    self.graphics()
                    return state, reward, done

        else:
            self.graphics()
            self.clock.tick(SPEED)
        #reward = 1/(abs(self.x1 - self.goal_pos[0])**2 + abs(self.y1 - self.goal_pos[1])**2 + 0.01)
            return self.get_state(),0, done

    def goal_reached(self):
        return abs(self.x1 - self.goal_pos[0]) <= 7 and abs(self.y1 - self.goal_pos[1]) <= 7

    def take_action(self,action):
        if action == 0:
            self.theta1 += self.step_size
            self.rotation = 1

        elif action == 1:
            self.theta1 -= self.step_size
            self.rotation = 0
        #
        # if self.theta1 > 180 or self.theta1 < 0:
        #     self.theta1 = 0

        self.transform(self.theta1)

    def plot(self, mean_scores, losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        fig = plt.figure()
        fig.add_axes()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        plt.title('Training...')
        plt.xlabel('Number of Games')
        ax1.plot(mean_scores)
        ax2.plot(losses)
        ax1.legend(['Rewards'])
        ax2.legend(['Losses'])


if __name__ == "__main__":

        r = Environment()
        remember = ReplayBuffer(1000, 1000000)

        model = DQL(1, 4,2)
        PATH = './models/pendulum.pth'
        losses = []
        NUM_EPISODES = 20000
        episode_reward = 0


        all_rewards = []
        max_reward = 0
        alpha = 0.99
        state = r.reset()
        j = 0

        while True:

            j += 1
            r.reset()
            print(r.get_state())
            for i in range(36):
                state = r.get_state()
                action = model.act(state)
                next_state, reward, done = r.step(action)
                model.compute_single_loss(state, action, reward, next_state, done)
                remember.add(state, action, reward, next_state, done)

                if reward > max_reward:
                    max_reward = reward
                    print('saving!...')

                all_rewards.append(reward)

                if done :
                    print("reset")
                    state = r.reset()
                    model.epsilon *= alpha
                    #all_rewards.append(episode_reward)
                    episode_reward = 0
                    model.n_games += 1


                if len(remember.memory) > remember.batch_size:
                    state, action, reward, next_state, done = zip(*remember.get_random())
                    loss = model.compute_batch_loss(state, action, reward, next_state, done)

                    losses.append(loss.item())

            if j % 5 == 0:
                model.save_model(PATH+str(j))
                print(f"Model saved at episode: {j}, # of games = {model.n_games}")
                r.plot(all_rewards, losses)









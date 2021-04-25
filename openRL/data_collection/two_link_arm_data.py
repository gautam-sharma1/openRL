import math
import pickle
import random
import pygame
from openRL.Agents import Agent

SPEED = 100


def forward_kinematics(length1, length2, theta1, theta2):

    x1 = length1*math.cos(math.radians(theta1))
    y1 = length1*math.sin(math.radians(theta1))

    x2 = x1 + length2*math.cos(math.radians(theta2))
    y2 = y1 + length2*math.sin(math.radians(theta2))
    return round(x1,2), round(y1,2), round(x2,2), round(y2,2)


class Environment(Agent):
    def __init__(self, length_link1 = 100, length_link2=100,origin=(200,200), robot_color=(0,255,0),
                 base_color=(255, 0, 0),goal_color=(255,50,255)):
        super().__init__()
        self.length_link1 = length_link1
        self.length_link2 = length_link2
        self.origin = origin
        self.x1 = origin[0] + length_link1
        self.y1 = origin[1]
        self.x2 = origin[0] + length_link1 + length_link2
        self.y2 = origin[1]
        self.theta1 = 0
        self.theta2 = 0
        self.goal_pos = (0,0)
        self.robot_color = robot_color
        self.base_color = base_color
        self.goal_color = goal_color
        self.step_size = 10
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((400, 400))
        pygame.display.set_caption('Two Link Arm Robot')

    def take_action(self,theta1, theta2) -> tuple: # alias for forward kinematics
        """
        Calculates forward kinematics in GLOBAL coordinates
        :param theta1:
        :param theta2:
        :return:
        """
        self.theta1 = theta1
        self.theta2 = theta2

        self.x1 = self.length_link1 * math.cos(math.radians(self.theta1))
        self.y1 = self.length_link1 * math.sin(math.radians(self.theta1))

        self.x2 = self.x1 + self.length_link2 * math.cos(math.radians(self.theta2))
        self.y2 = self.y1 + self.length_link2 * math.sin(math.radians(self.theta2))

        return round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)

    # TODO: Write a transform function
    # def update(self):
    #     self.goal_pos = (self.x2+self.origin[0], -self.y2+self.origin[1])

    def get_state(self, *args, **kvargs):
        return self.theta1,self.theta1

    def terminate(self, *args, **kvargs):
        pass

    def graphics(self,goal_pos):
        self.display.fill((0, 0, 0))
        self.draw_players(goal_pos)
        pygame.display.flip()

    def reset(self):
        self.x1 = self.origin[0] + self.length_link1
        self.y1 = self.origin[1]
        self.x2 = self.origin[0] + self.length_link1 + self.length_link2
        self.y2 = self.origin[1]
        self.theta1 = 0
        self.theta2 = 0
        #self.graphics()

    def step(self,theta1,theta2,goal_pos):

        self.reset()
        self.graphics(goal_pos)
        for steps in range(1,int(theta1)+1,self.step_size):
            self.take_action(self.theta1+self.step_size,self.theta2)
            print(self.theta1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            #self.graphics(goal_pos)

        for steps in range(1,int(theta2)+1,self.step_size):
            self.take_action(self.theta1,self.theta2 + self.step_size)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            #self.graphics(goal_pos)

        print(self.x2,self.y2)


    def draw_goal(self, goal_pos):
        self.display.fill((0, 0, 0))
        pygame.draw.circle(self.display, self.goal_color, (int(goal_pos[0]) + self.origin[0], int(-goal_pos[1]) + self.origin[1]), 10, 2)
        pygame.display.flip()

    def draw_players(self,goal_pos):
        pygame.draw.line(self.display,self.robot_color,self.origin,(self.x1+self.origin[0],-self.y1+self.origin[1]),10)
        pygame.draw.line(self.display, self.robot_color, (self.x1+self.origin[0],-self.y1+self.origin[1]), (self.x2+self.origin[0], -self.y2+self.origin[1]), 10)
        pygame.draw.circle(self.display,self.base_color,self.origin,2,1)
        pygame.draw.circle(self.display, self.base_color, (int(self.x1+self.origin[0]),int(-self.y1+self.origin[1])), 2, 1)
        pygame.draw.circle(self.display, self.goal_color, (int(goal_pos[0]) + self.origin[0], int(-goal_pos[1] )+ self.origin[1]), 10, 2)


if __name__ == "__main__":

    LENGTH = 100
    ITERATIONS = 1000
    # print(forward_kinematics(100, 100, 0, 0))
    # print(forward_kinematics(100, 100,0,90))
    # print(forward_kinematics(100, 100,45,45))
    input_list = []
    output_list = []

    for i in range(ITERATIONS):
        theta1 = random.randint(0, 180)
        theta2 = random.randint(0, 180)
        input_list.append(forward_kinematics(LENGTH, LENGTH, theta1, theta2))
        output_list.append([theta1,theta2])

    assert len(input_list) == ITERATIONS
    with open('./files/2link_pendulum_input_new.pkl', 'wb') as f:
        pickle.dump(input_list, f)

    assert len(output_list) == ITERATIONS
    with open('./files/2link_pendulum_output_new.pkl', 'wb') as f:
        pickle.dump(output_list, f)

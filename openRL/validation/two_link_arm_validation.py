#########################################################
# VALIDATION SCRIPT  #
# TWO LINK ARM #
# @author Gautam Sharma
#########################################################
__author__ = "Gautam Sharma"

import pickle
from openRL.AI import Netleaky
import pygame
import torch
import random
import math
import matplotlib.pyplot as plt
from openRL.data_collection.two_link_arm_data import Environment

def forward_kinematics(length1, length2, theta1, theta2):

    x1 = length1*math.cos(math.radians(theta1))
    y1 = length1*math.sin(math.radians(theta1))

    x2 = x1 + length2*math.cos(math.radians(theta2))
    y2 = y1 + length2*math.sin(math.radians(theta2))
    return round(x1,2), round(y1,2), round(x2,2), round(y2,2)

SPEED = 100
START = 10
NUM_ITERATIONS = 200
if __name__ == "__main__":
    LENGTH = 100
    r = Environment()
    net = Netleaky(4, 2)
    PATH = './models/net.pthb965'
    with open('./files/2link_pendulum_input_new.pkl', 'rb') as f:
        input_list = pickle.load(f)

    with open('./files/2link_pendulum_output_new.pkl', 'rb') as f:
        output_list = pickle.load(f)

    predicted_list_theta1 = []
    predicted_list_theta2 = []
    goal_pos_list = []
    predicted_end_effector_pose = []
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    with torch.no_grad():
        NUM_TRAJECTORY = 20
        j = 0
        for i in range(START, NUM_ITERATIONS):

            theta1 = random.randint(0, 180)
            theta2 = random.randint(0, 180)
            expected_end_effector_pose = forward_kinematics(LENGTH, LENGTH, theta1, theta2)
            expected_theta = [theta1, theta2]
            predicted_theta = net(torch.FloatTensor(expected_end_effector_pose))
            predicted_list_theta1.append(predicted_theta[0] * 90)
            predicted_list_theta2.append(predicted_theta[1] * 90)
            print(
                f"Expected Pose = {expected_end_effector_pose}, Expected Theta = {expected_theta}, Predicted Theta = {predicted_theta * 90}")
            #r.draw_goal(expected_end_effector_pose)

            goal_pos = expected_end_effector_pose[2],expected_end_effector_pose[3]
            x1,y1, x2,y2 = forward_kinematics(100,100,expected_theta[0],expected_theta[1])

            r.step(predicted_theta[0] * 90, predicted_theta[1] * 90, goal_pos)

            # r.reset()

            x1p,y1p, x2p,y2p = r.take_action(predicted_theta[0] * 90, predicted_theta[1] * 90)

            goal_pos_list.append([x2,y2])
            predicted_end_effector_pose.append([x2p,y2p])
            print(f"Expected end effector pose:{x2,y2}, Actual = {x2p,y2p}")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            #pygame.time.wait(1000)




    output_list_subset = output_list[START:NUM_ITERATIONS]

    fig, (ax1, ax2) = plt.subplots(2, 1)


    ax1.plot([t for t in range(NUM_ITERATIONS-START)], [x[0] for x in output_list_subset], 'r-',
             [t for t in range(NUM_ITERATIONS-START)], predicted_list_theta1,'g-.')


    ax2.plot([t for t in range(NUM_ITERATIONS-START)], [x[1] for x in output_list_subset], 'r-',
             [t for t in range(NUM_ITERATIONS-START)],
             predicted_list_theta2, 'g-.')

    ax1.legend(['Expected theta 1', 'Predicted theta 1'])
    ax1.set_title("Predicted and Expected Theta 1")
    # x1.set_xlabel("# Episode")
    ax1.set_ylabel("Angle (Degrees)")

    ax2.legend(['Expected theta 2', 'Predicted theta 2'])
    ax2.set_title("Predicted and Expected Theta 2")
    ax2.set_ylabel("Angle (Degrees)")




    fig.savefig("./figures/twoLinkArmTheta.png", dpi=300)


    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot([t for t in range(NUM_ITERATIONS - START)], [x[0] for x in predicted_end_effector_pose], 'g-.',

             [t for t in range(NUM_ITERATIONS - START)], [x[0] for x in goal_pos_list], 'r-')

    ax2.plot([t for t in range(NUM_ITERATIONS - START)], [x[1] for x in predicted_end_effector_pose], 'g-.',
             [t for t in range(NUM_ITERATIONS - START)], [x[1] for x in goal_pos_list], 'r-')
    ax1.set_ylabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax1.legend(['Actual end effector x coordinate', 'Goal position x coordinate'])

    ax2.legend(['Actual end effector y coordinate', 'Goal position y coordinate'])

    fig.savefig("./figures/twoLinkArmPose.png", dpi=300)

    plt.show()

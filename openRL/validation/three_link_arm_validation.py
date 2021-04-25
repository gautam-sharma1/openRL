#########################################################
# VALIDATION SCRIPT  #
# THREE LINK ARM #
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
from openRL.data_collection.three_link_arm_data import Environment


def forward_kinematics(length1, length2, length3, theta1, theta2, theta3):

    x1 = length1 * math.cos(math.radians(theta1))
    y1 = length1 * math.sin(math.radians(theta1))

    x2 = x1 + length2 * math.cos(math.radians(theta2))
    y2 = y1 + length2 * math.sin(math.radians(theta2))

    x3 = x2 + length3 * math.cos(math.radians(theta3))
    y3 = y2 + length3 * math.sin(math.radians(theta3))

    return round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(x3, 2), round(y3, 2)


SPEED = 100
NUM_ITERATIONS = 200
START = 10
if __name__ == "__main__":
    LENGTH1 = 100
    LENGTH2 = 50
    LENGTH3 = 50
    r = Environment()
    net = Netleaky(6, 3)
    PATH = './models/net.pthc241'
    with open('./files/3link_pendulum_input_new.pkl', 'rb') as f:
        input_list = pickle.load(f)

    with open('./files/3link_pendulum_output_new.pkl', 'rb') as f:
        output_list = pickle.load(f)

    predicted_list_theta1 = []
    predicted_list_theta2 = []
    predicted_list_theta3 = []
    goal_pos_list = []
    predicted_end_effector_pose = []

    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    with torch.no_grad():
        NUM_TRAJECTORY = 20
        j = 0
        for i in range(START, NUM_ITERATIONS):

            theta1 = random.randint(0, 180)
            theta2 = random.randint(0, 180)
            theta3 = random.randint(0, 180)

            expected_end_effector_pose = forward_kinematics(LENGTH1, LENGTH2, LENGTH3, theta1, theta2, theta3)
            expected_theta = [theta1, theta2, theta3]

            predicted_theta = net(torch.FloatTensor(expected_end_effector_pose))

            predicted_list_theta1.append(float(predicted_theta[0] * 90))
            predicted_list_theta2.append(float(predicted_theta[1] * 90))
            predicted_list_theta3.append(float(predicted_theta[2] * 90))

            print(
                f"Expected Pose = {expected_end_effector_pose}, Expected Theta = {expected_theta}, Predicted Theta = {predicted_theta * 90}")

            goal_pos = expected_end_effector_pose[4], expected_end_effector_pose[5]

            x1, y1, x2, y2, x3, y3 = forward_kinematics(100, 50, 50, expected_theta[0], expected_theta[1],
                                                        expected_theta[2])

            r.step(predicted_theta[0] * 90, predicted_theta[1] * 90, predicted_theta[2] * 90,
                   goal_pos)  # for simulation

            x1p, y1p, x2p, y2p, x3p, y3p = r.take_action(predicted_theta[0] * 90, predicted_theta[1] * 90,
                                                         predicted_theta[2] * 90)

            goal_pos_list.append([x3,y3])
            predicted_end_effector_pose.append([x3p,y3p])

            print(f"Expected end effector pose:{x3, y3}, Actual = {x3p, y3p}")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
    output_list_subset = output_list[START:NUM_ITERATIONS]
    print(predicted_list_theta1)
    print([x[0] for x in output_list_subset])
    print(len(output_list_subset))
    print(len(predicted_list_theta1))
    print(len(range(NUM_ITERATIONS-START)))

    fig,(ax1,ax2,ax3) = plt.subplots(3,1)

    ax1.plot([t for t in range(NUM_ITERATIONS-START)], [x[0] for x in output_list_subset], 'r-',
             [t for t in range(NUM_ITERATIONS-START)], predicted_list_theta1, 'g-.')

    ax2.plot([t for t in range(NUM_ITERATIONS-START)], [x[1] for x in output_list_subset], 'r-',
             [t for t in range(NUM_ITERATIONS-START)],predicted_list_theta2, 'g-.')

    ax3.plot([t for t in range(NUM_ITERATIONS-START)], [x[2] for x in output_list_subset], 'r-',
             [t for t in range(NUM_ITERATIONS-START)],predicted_list_theta3, 'g-.')

    ax1.legend(['Expected theta 1', 'Predicted theta 1'])
    ax1.set_title("Predicted and Expected Theta 1")
    #x1.set_xlabel("# Episode")
    ax1.set_ylabel("Angle (Degrees)")

    ax2.legend(['Expected theta 2', 'Predicted theta 2'])
    ax2.set_title("Predicted and Expected Theta 2")
    ax2.set_ylabel("Angle (Degrees)")
    #ax2.set_xlabel("# Episodes")

    ax3.legend(['Expected theta 3', 'Predicted theta 3'])
    ax3.set_title("Predicted and Expected Theta 3")
    ax3.set_ylabel("Angle (Degrees)")
    #ax3.set_xlabel("# Episodes")


    fig.savefig("./figures/threeLinkArmTheta.png", dpi=300)

    fig,(ax1,ax2) = plt.subplots(2,1)

    ax1.plot([t for t in range(NUM_ITERATIONS-START)],[x[0] for x in predicted_end_effector_pose], 'g+',

             [t for t in range(NUM_ITERATIONS-START)],[x[0] for x in goal_pos_list], 'ro')

    ax2.plot([t for t in range(NUM_ITERATIONS-START)],[x[1] for x in predicted_end_effector_pose], 'g+',[t for t in range(NUM_ITERATIONS-START)],[x[1] for x in goal_pos_list], 'ro')
    ax1.set_ylabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax1.legend(['Actual end effector x coordinate', 'Goal position x coordinate'])

    ax2.legend(['Actual end effector y coordinate', 'Goal position y coordinate'])

    fig.savefig("./figures/threeLinkArmPose.png", dpi=300)

    plt.show()



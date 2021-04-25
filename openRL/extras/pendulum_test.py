import math
import random
import pickle


def forward_kinematics(length, theta):

    x1 = length*math.cos(math.radians(theta))
    y1 = length*math.sin(math.radians(theta))
    return round(x1,2), round(y1,2)


print(forward_kinematics(100,120))
print(forward_kinematics(100,0))
print(forward_kinematics(100,45))

input_list = []
output_list = []
LENGTH = 100
# for i in range(1000):
#     theta = random.randint(0, 90)
#     input_list.append(forward_kinematics(LENGTH, theta))
#     output_list.append(theta)
#
#
# assert len(input_list) == 1000
# with open('input_pendulum_new.pkl', 'wb') as f:
#     pickle.dump(input_list, f)
#
# assert len(output_list) == 1000
# with open('output_pendulum_new.pkl', 'wb') as f:
#     pickle.dump(output_list, f)


#### EVAL ###
from openRL.AI import Netleaky
import torch
import matplotlib.pyplot as plt

NUM_ITERATIONS = 200
PATH = './models/net.pthlast4953'
with open('input_pendulum_new.pkl', 'rb') as f:
    input_list = pickle.load(f)

with open('output_pendulum_new.pkl', 'rb') as f:
    output_list = pickle.load(f)

predicted_list_theta = []
net = Netleaky(2, 1)
net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

with torch.no_grad():
    for i in range(0, NUM_ITERATIONS):
        expected_end_effector_pose = input_list[i]
        expected_theta = output_list[i]
        predicted_theta = net(torch.FloatTensor(expected_end_effector_pose))
        predicted_list_theta.append(predicted_theta*90)
        print(f"Expected Pose = {expected_end_effector_pose}, Expected Theta = {expected_theta}, Predicted Theta = {predicted_theta*90}")

output_list_subset = output_list[:NUM_ITERATIONS]

plt.plot(output_list_subset)
plt.plot(predicted_list_theta)
#
# plt.legend(['Expected theta 1','Predicted theta 1'])
# plt.title('Theta 1 error')
plt.show()
plt.waitforbuttonpress()

#####

# IN PROGRESS

####

from openRL.pendulum import Environment
from openRL.ReplayBuffer import ReplayBuffer
from openRL.AI import DQL,Net,NNet
import torch
import torch.utils.data as Data
import pickle
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler

#######
import matplotlib
import matplotlib.pyplot as plt


import numpy as np
import imageio


###########


with open('input.pkl', 'rb') as f:
    output_list = pickle.load(f)


with open('output.pkl', 'rb') as f:
    input_list = pickle.load(f)

import pandas as pd


df1 = pd.DataFrame(input_list)
df2 = pd.DataFrame(output_list)
df1.to_csv('input.csv', index=False)
df2.to_csv('output.csv', index=False)
#
# X_train = df1.iloc[:,:].values
# y_train = df2.iloc[:,:].values
# print(len(X_train))
# sc = MinMaxScaler()
# sct = MinMaxScaler()
#
# X_train=sc.fit_transform(X_train.reshape(-1,2))
# #y_train =sct.fit_transform(y_train.reshape(-1,2))
# print(X_train)

# view data
# plt.figure(figsize=(10,4))
# plt.scatter(input_list, output_list, color = "orange")
# plt.title('Regression Analysis')
# plt.xlabel('Independent varible')
# plt.ylabel('Dependent varible')
# plt.show()
# #######
# input()

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_list, labels):
        'Initialization'
        self.labels = labels
        self.input = input_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.input)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.FloatTensor(self.input[index])

        y = torch.FloatTensor(self.labels[index])

        return X, y


# Parameters
params = {'batch_size': 1,
          'shuffle': True}
max_epochs = 100

# Datasets
input_data = input_list

labels = output_list

# Generators
training_set = Dataset(input_data, labels)

training_generator = torch.utils.data.DataLoader(training_set, **params)
PATH = './models/imitation/net.pthnew'
net = Net(2,2)
#net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

loss = 0
min_loss = 100000
# train the network

for t in range(1000):
    for local_batch, local_labels in training_generator:

        prediction = net(local_batch)  # input x and predict based on x

        loss = loss_func(prediction, local_labels)  # must be (1. nn output, 2. target)
        if loss < min_loss:
            torch.save(net.state_dict(), PATH+str(t))
            min_loss = loss

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    plt.plot(t, loss.detach().numpy())

    print(f"Episode #{t} : Loss {loss}")

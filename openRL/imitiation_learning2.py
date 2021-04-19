#####

# THIS FILE NOT NEEDED

####


from openRL.pendulum import Environment
from openRL.ReplayBuffer import ReplayBuffer
from openRL.AI import DQL,Net,NNet
import torch
import torch.utils.data as Data
import pickle
from torchvision import transforms
from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt                        # so we can add to plot
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, BayesianRidge)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.neural_network import MLPClassifier       # learning algorithm
from sklearn.metrics import confusion_matrix           # confusion matrix
from sklearn.model_selection import GridSearchCV       # optimize over parameters
from sklearn.decomposition import PCA                  # PCA package
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import pandas as pd

#######



with open('input.pkl', 'rb') as f:
    output_list = pickle.load(f)


with open('output.pkl', 'rb') as f:
    input_list = pickle.load(f)





df1 = pd.DataFrame(input_list)
df2 = pd.DataFrame(output_list)


X = df1.iloc[:,:].values
y = df2.iloc[:,:].values
sc = StandardScaler()
sc.fit(X)                                       # compute the required transformation
X_train_std = sc.transform(X)                   # apply to the training data



#


clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
print(clf.score(X, y))
print(clf.predict([250,250]))
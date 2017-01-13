import numpy as np
import itertools
from Features import features
# Load the data
X = np.load('X_Data.npy')
Y = np.load('Y_Data.npy')
# Find the unique elements
id = np.unique(Y)

# Initialisation
XP = []
YP = []
XN = []
YN = []

x = np.load('X_Train.npy')

x0 = x[0]

print(x0[0:1600].shape,x0[1600:].shape)

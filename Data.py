import numpy as np
import itertools
import Image_process
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


for item in id:
    # Count the number of items in given class
    count = sum(Y==item)
    # Divide the set into positive and negative examples
    xp = X[Y==item]
    xn = X[Y!=item]
    # Get all the combination of items in a given class
    combin = list(itertools.combinations(range(count),2))
    # Now get all the vectors using the above combination
    for ele in combin:
        # Positive examples
        N = np.array(np.hstack(abs(xp[ele[0]]-xp[ele[1]])))
        # N = features(xp[ele[0]],xp[ele[1]])
        XP.append(N)
        YP.append(np.array([1]))
        # Negative examples
        M = np.array(np.hstack(abs(xp[ele[0]]-xn[np.random.randint(xn.shape[0],size=1)[0]])))
        # M = features(xp[ele[0]],xn[np.random.randint(xn.shape[0],size=1)[0]])
        XN.append(M)
        YN.append(np.array([0]))


X = np.vstack((XP,XN))
Y = np.vstack((YP,YN))
p = np.random.permutation(range(np.shape(X)[0]))
X_Train, Y_Train = X[p],Y[p]

np.save('X_Train',X_Train)
np.save('Y_Train',Y_Train)

print(X.shape)
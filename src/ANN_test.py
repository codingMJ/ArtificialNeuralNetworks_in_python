import numpy as np
from numpy.linalg import norm
from ANN import ANN
from Trainer import Trainer

#_________________________________
#  DATA BEFORE REGULARIZATION
#_________________________________

# X = np.array(([3,5], [5,1], [10,2]), dtype = float)
# y = np.array(([75], [82], [93]), dtype = float)

# Scaling
# X = X/np.amax(X, axis = 0)
# y = y/100 # max test score is 100
# print(X)
# print(y)

# NN = ANN()
#  
# T = Trainer(NN)
# T.train(X, y)

# yHat = NN.forward(X)
# print(yHat)
# print(y)



#_________________________________
#  DATA AFTER REGULARIZATION
#_________________________________

#Training Data:
trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
trainY = np.array(([75], [82], [93], [70]), dtype=float)

#Testing Data:
testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

#Normalize:
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100 #Max test score is 100

#Normalize:
testX = testX/np.amax(testX, axis=0)
testY = testY/100 #Max test score is 100

#Train network with new data:
NN = ANN(Lambda=0.0001)

T = Trainer(NN)
T.train(trainX, trainY, testX, testY)

  
# yHat = NN.forward(X)
# print(yHat)
# print(y)

# print(NN.W1)
# print()
# print(NN.W2)
# print()
# print(NN.W1**2)
# print()
# print(NN.W2**2)
# print()
# print(sum(NN.W1**2))
# print()
# print(sum(NN.W2**2))
# print()
# print(sum(NN.W1**2) + sum(NN.W2**2))
# print()
# print(X.shape[0])
# d = NN.d(X, y)


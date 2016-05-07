import numpy as np

class ANN(object):
    def __init__(self, Lambda=0):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
        # Set weights
        np.random.seed(2)
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)
        
        self.W1[0][0] = 0.73
        self.W1[0][1] = 0.86
        self.W1[0][2] = 0.74   
        
        self.W1[1][0] = 0.5
        self.W1[1][1] = 0.23
        self.W1[1][2] = 0.03
        
        self.W2[0] = 0.99;
        self.W2[1] = 0.07;
        self.W2[2] = 0.69;
        
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
        
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    
    def sigmoidPrime(self, z):
        return np.exp(-z)/((1 + np.exp(-z))**2)
    
    #_________________________________
    #  costFunction and costFunctionPrime
    #
    #  BEFORE REGULARIZATION
    #_________________________________
    
#     def costFunction(self, X, y):
#         #Compute cost for given X,y, use weights already stored in class.
#         self.yHat = self.forward(X)
#         J = 0.5 * sum((y - self.yHat)**2)
#         return J 
#     
#     
#     def costFunctionPrime(self, X, y):
#         #Compute derivative with respect to W and W2 for a given X and y:
#         self.yHat = self.forward(X)
#         
#         delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
#         dJdW2 = np.dot(self.a2.T, delta3)
#         
#         delta2 = np.multiply(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
#         dJdW1 = np.dot(X.T, delta2)
# 
#         return dJdW1, dJdW2
    
    
    #_________________________________
    #  costFunction and costFunctionPrime
    #
    #  AFTER REGULARIZATION
    #_________________________________
    
    
    #Need to make changes to costFunction and costFunctionPrim:
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(sum(self.W1**2)+sum(self.W2**2))
        return J
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
    
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
    
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
    
        return dJdW1, dJdW2
    
    
    def d(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        d = np.multiply(delta3, self.W2.T)
        delta2 = np.multiply(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return delta3, dJdW1, d, self.sigmoidPrime(self.z2), delta2
    
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize

        self.W1 = np.reshape(params[W1_start : W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end : W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
         
        
#     def computeNumericalGradient(self, N, X, y):
#         paramsInitial = N.getParams()
#         numGrad = np.zeros(paramsInitial.shape)
#         perturb = np.zeros(paramsInitial.shape)
#         e = 1e-4
#         
#         for p in range(len(paramsInitial)):
#             # Set perturbation vector
#             perturb[p] = e
#             N.setParams(paramsInitial + perturb)
#             loss2 = N.costFunction(X, y)
#             
#             N.setParams(paramsInitial - perturb)
#             loss1 = N.costFunction(X, y)
#             
#             # Compute numerical gradient
#             numGrad[p] = (loss2 - loss1) / (2 * e)
#             
#             # Return the value we changed back to zero
#             perturb[p] = 0
#         
#         # Return Params to original value
#         N.setParams(paramsInitial)
#         
#         return numGrad  
        
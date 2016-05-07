from scipy import optimize


class Trainer(object):
    def __init__(self, N):
        # Make local reference to Neural Network
        self.N = N
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad
    
    
#_________________________________
#  FUNCTIONS BEFORE REGULARIZATION
#_________________________________

#     def callBackF(self, params):
#         self.N.setParams(params)
#         self.J.append(self.N.costFunction(self.X, self.y))
    
#     def train(self, X, y):
#         # Make internal variable for callBack function 
#         self.X = X
#         self.y = y
#         
#         # make empty list to store costs
#         self.J = []
#         
#         params0 = self.N.getParams()
# 
#         
#         options = {'maxiter' : 200, 'disp': True}
#         _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', \
#                                 args = (X, y), options =  options, callback = self.callBackF)
#         
# #         self.N.setParams(_res.x)
#         self.optimizationResults = _res


#_________________________________
#  FUNCTIONS AFTER REGULARIZATION
#_________________________________

    def callBackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def train(self, trainX, trainY, testX, testY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        #Make empty list to store training costs:
        self.J = []
        self.testJ = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callBackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
        
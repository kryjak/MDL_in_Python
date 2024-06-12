import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

class BatchUtils(object):
    def __init__(self) -> None:
        pass
    
    def get_batches(self, X, Y, batch_size, return_targets: bool = False):
        N = len(X)
        n_batches = int(np.ceil(N / batch_size))
        if return_targets:
            T = self.encode_targets(Y)
            batches = [[X[i*batch_size: (i+1)*batch_size], Y[i*batch_size: (i+1)*batch_size], T[i*batch_size: (i+1)*batch_size]] for i in range(n_batches)] 
        else:
            batches = [[X[i*batch_size: (i+1)*batch_size], Y[i*batch_size: (i+1)*batch_size]] for i in range(n_batches)] 
        return batches

class MultiClassUtils(object):
    def __init__(self) -> None:
        pass

    def encode_targets(self, Y):
        n_classes = len(set(Y))
        T = np.zeros((len(Y), n_classes), dtype=np.int8)
        for ii in range(len(Y)):
            T[ii, Y[ii]] = 1

        return T
    
    def predict(self, pY):
        if pY.ndim > 1:
            return np.argmax(pY, axis=1)
        else:
            return np.argmax(pY)
            
    def cross_entropy_loss(self, pY, T):
        return -np.sum(T * np.log(pY))

    def classification_rate(self, pY, Y):
        preds = self.predict(pY)
        return np.mean(preds == Y)
    

class LogisticRegressionUtils(object):
    def __init__(self) -> None:
        pass
    
    def forward_pass(self, X, W, b):
        pY = sp.softmax(X.dot(W) + b, axis=1)
        return pY
    
    def derivative_W(self, X, pY, T):
        return X.T.dot(pY - T)
    
    def derivative_b(self, pY, T):
        return np.sum(pY - T, axis=0)
    
class ActivationFunctions(object):
    def __init__(self) -> None:
        pass

    def ReLU(self, x):
        return x * (x > 0)
    
    def GeLU(self, x):
        return x/2 * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))
    
    def grad_tanh(self, x):
        return 1 - x*x
    
    def grad_ReLU(self, x):
        return np.heaviside(x, 0)
    
    # https://arxiv.org/pdf/2305.12073
    def grad_GeLU(self, x):
        return 1/2 * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))) + x/2 / np.cosh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))**2 * np.sqrt(2/np.pi)*(1 + 3*0.044715*x**2)
    
class ANNUtils(ActivationFunctions):
    # X - input matrix, size NxD
    # Z - hidden layer, size NxM
    # pY - output layer, size NxK
    # T - target one-hot encoded as an indicator matrix, size NxK
    # W2 - weight matrix of the second layer
    def derivative_W1(self, X, dZ, W2, pY, T):
        return X.T.dot((pY - T).dot(W2.T) * dZ)
    def derivative_b1(self, dZ, W2, pY, T):
        return np.sum((pY - T).dot(W2.T) * dZ, axis=0)
    def derivative_W2(self, Z, pY, T):
        return Z.T.dot(pY - T)
    def derivative_b2(self, pY, T):
        return np.sum(pY - T, axis=0)
    
    def feedforward(self, X, W1, b1, W2, b2):
        a = X.dot(W1) + b1
        Z = self.act_fun(a)
        alpha = Z.dot(W2) + b2
        if alpha.ndim > 1:
            pY = sp.softmax(alpha, axis=1)
        else:
            pY = sp.softmax(alpha)
        return Z, pY

class MultiClassLogisticRegression(LogisticRegressionUtils, MultiClassUtils, BatchUtils):
    def __init__(self) -> None:
        pass

    def fit(self, X, Y, epochs=10**3, learning_rate=1e-5, reg=0):
        N, D = X.shape
        # initialise weights and biases:
        n_classes = len(set(Y))
    
        self.W = np.random.normal(size=(D, n_classes)) / np.sqrt(D)
        self.b = np.zeros((1, n_classes))
    
        cost = []
        error = []
    
        T = self.encode_targets(Y)
        for ii in range(epochs):
            pY = self.forward_pass(X, self.W, self.b)

            grad_W = self.derivative_W(X, pY, T) + reg*self.W
            grad_b = self.derivative_b(pY, T) + reg*self.b
            self.W -= learning_rate * grad_W
            self.b -= learning_rate * grad_b

            if ii % 20 == 0:
                cost.append(self.cross_entropy_loss(pY, T))
                error_rate = 1 - self.classification_rate(self.predict(pY), Y)

                print(f'Cost at epoch {ii}: {cost[-1]}, training error rate: {error_rate}')

        plt.plot(cost)
        plt.show()
    
        print(f'Error rate: {error_rate}')
        
    def classify(self, X):
        return self.predict(self.forward_pass(X, self.W, self.b))
        
    def evaluate(self, X_test, Y_test):
        preds = self.classify(X_test)
        T = self.encode_targets(Y_test)
        error_rate = 1 - self.classification_rate(preds, T)
        return error_rate   
    
class ANNOneHiddenLayer(ANNUtils, MultiClassUtils, BatchUtils):
    def __init__(self, M:int = 10, activation_function:str = 'tanh') -> None:
        super().__init__()

        self.M = M
        self.activation_function = activation_function
        if self.activation_function == 'tanh':
            self.act_fun = np.tanh
            self.dZ = self.grad_tanh
        elif self.activation_function == 'ReLU':
            self.act_fun = self.ReLU
            self.dZ = self.grad_ReLU
        elif self.activation_function == 'GeLU':
            self.act_fun = self.GeLU
            self.dZ = self.grad_GeLU

    def fit(self, X, Y, epochs=10**3, learning_rate=1e-5, reg=0, verbose: bool = False):
        N, D = X.shape
        # initialise weights and biases:
        n_classes = len(set(Y))
    
        self.W = np.random.normal(size=(D, n_classes)) / np.sqrt(D)
        self.b = np.zeros((1, n_classes))
    
        cost = []
        error = []
    
        T = self.encode_targets(Y)

        # randomly initialize weights
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, n_classes) / np.sqrt(self.M)
        self.b2 = np.zeros(n_classes)

        for ii in range(epochs):
            Z, pY = self.feedforward(X, self.W1, self.b1, self.W2, self.b2)

            self.W1 -= learning_rate * (self.derivative_W1(X, self.dZ(Z), self.W2, pY, T) + reg*self.W1)
            self.b1 -= learning_rate * (self.derivative_b1(self.dZ(Z), self.W2, pY, T) + reg*self.b1)
            self.W2 -= learning_rate * (self.derivative_W2(Z, pY, T) + reg*self.W2)
            self.b2 -= learning_rate * (self.derivative_b2(pY, T) + reg*self.b2)


            if ii % 20 == 0:
                cost.append(self.cross_entropy_loss(pY, T))
                error_rate = 1 - self.classification_rate(pY, Y)
                if verbose:
                    print(f'Cost at epoch {ii}: {cost[-1]}, training error rate: {error_rate}')

        plt.plot(cost)
        plt.show()
    
        print(f'Training error rate: {error_rate}')
        
    def classify(self, X):
        _, pY = self.feedforward(X, self.W1, self.b1, self.W2, self.b2)
        return self.predict(pY)
        
    def evaluate(self, X_test, Y_test):
        T_pred = self.classify(X_test)
        error_rate = 1 - np.mean(T_pred == Y_test)
        return T_pred, error_rate   
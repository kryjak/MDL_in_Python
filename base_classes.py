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

class LogisticRegressionUtils(object):
    def __init__(self) -> None:
        pass
    
    def classification_rate(self, pY, Y):
        preds = self.predict(pY)
        return np.mean(preds == Y)
    
    def forward_pass(self, X, W, b):
        pY = sp.softmax(X.dot(W) + b, axis=1)
        return pY
    
    def derivative_W(self, X, pY, T):
        return X.T.dot(pY - T)
    
    def derivative_b(self, pY, T):
        return np.sum(pY - T, axis=0)

class MultiClassLogisticRegression(LogisticRegressionUtils, MultiClassUtils, BatchUtils):
    def __init__(self) -> None:
        self.W = None
        self.b = None

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
        preds = self.classify(X_test, self.W, self.b)
        T = self.encode(self.encode_targets(Y_test))
        error_rate = 1 - self.classification_rate(preds, T)
        return error_rate   
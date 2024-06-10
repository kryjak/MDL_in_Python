import numpy as np
import scipy.special as sp

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

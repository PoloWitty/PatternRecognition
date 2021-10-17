from sklearn.utils import shuffle
import numpy as np
def data_generator(mean1, cov1, mean2, cov2, num,seed=0):
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean1, cov1, num//2)#shape=(num//2,2)
    y = np.full(num//2, 1, dtype=np.float)#shape=(num//2,)
    
    np.random.seed(seed+1)
    X_ = np.random.multivariate_normal(mean2, cov2, num-num//2)
    y_ = np.full(num-num//2, -1, dtype=int)
    
    X1 = np.concatenate((X, X_), axis=0)
    y1 = np.concatenate((y, y_))

    return shuffle(X1, y1)
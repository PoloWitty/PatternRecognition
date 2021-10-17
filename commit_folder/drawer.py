import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def draw(data,w,marker='o'):
    if w[2] == 0:
        return
    X=data[0];y=data[1]
    # _ , ax=plt.subplots(1,1)
    ax=plt.gca()
    ax.scatter(X[y==1,0],X[y==1,1],c='r',marker=marker)
    ax.scatter(X[y==-1,0],X[y==-1,1],c='b',marker=marker)
    x1_=np.linspace(np.min(X),np.max(X),X.shape[0])
    x2_=-w[1]/w[2]*x1_ - w[0]/w[2]

    ax.plot(x1_,x2_)
    ax.set(xlabel='$x_1$',ylabel='$x_2$')
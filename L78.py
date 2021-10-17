# %%
import cvxopt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_gen import *
SEED=0
# %%[markdown]
## T1
### Primal-SVM

# %%

class Primal_SVM:
    def __init__(self,transform_fn=None):
        self.transform_fn=transform_fn
    def fit(self,data):
        X=data[0]
        if self.transform_fn:
            X=self.transform_fn.fit_transform(X)
        y=data[1]
        N,d=X.shape # N: the num of samples  d: 不增广1的时候的维数
        X=np.hstack((np.ones((N,1)),X))
        y=y.reshape((N,1))

        Q=np.array((np.hstack((np.zeros((1+d,1)),np.vstack([np.zeros((1,d)),np.eye(d)])))),dtype=np.float)
        p=np.zeros((d+1,1),dtype=np.float)
        A=-y*X
        c=np.full(N,-1.0).reshape((N,1))#注意类型要是小数类型, 不能是整数

        Q=cvxopt.matrix(Q)
        p=cvxopt.matrix(p)
        A=cvxopt.matrix(A)
        c=cvxopt.matrix(c)

        solution=cvxopt.solvers.qp(Q,p,A,c)
        weight=solution['x']
        self.b=weight[0]
        self.w=np.array(weight[1:]).flatten()

    def decision_fn(self,X):
        if self.transform_fn:
            X=self.transform_fn.fit_transform(X)
        return self.b+X@self.w

    def predict(self,X):
        return np.sign(self.decision_fn(X))
    
    def eval(self,X,y):
        pred=self.predict(X)
        mistake_indices = np.where(pred!=y)[0]
        accuracy = (X.shape[0]-len(mistake_indices))/X.shape[0]
        return accuracy
# X=np.array([[1,1],[2,2],[2,0],[0,0],[1,0],[0,1]])
# y=np.array([1]*3+[-1]*3,dtype=np.float)
# data=(X,y)
# model=Primal_SVM()
# model.fit(data)
# model.eval(data[0],data[1])
# %%
# SGD法求SVM的解
# X=np.array([[1,1],[2,2],[2,0],[0,0],[1,0],[0,1]])
# y=np.array([1]*3+[-1]*3)
# X=np.hstack((np.ones((6,1)),X))
# w0=np.array([0,0,0])
# w=w0
# lr=0.1
# for e in range(100):
#     g=((1-y*(X@w)>0)*(-y*X.T)).T
#     print('epoch ',e)
#     for i in range(6):
#         w=w-lr*g[i]
#         print(w)

# %%[markdown]
### dual SVM

# %%
class Dual_SVM:
    def __init__(self,epsilon=1e-9,transform_fn=None):
        self.epsilon=epsilon
        self.transform_fn=transform_fn

    def fit(self,data):
        X=data[0]#注意此处的X也是不增广1的X
        if self.transform_fn:
            X=self.transform_fn.fit_transform(X)
        y=data[1]
        N,d=X.shape

        Q=np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Q[i,j]=y[i]*y[j]*X[i,:]@X[j,:].T

        Q=np.array((Q),dtype=np.float)
        p=np.full((N,1),-1,dtype=np.float)
        A=-np.eye(N,dtype=np.float)#ppt上的公式有问题
        c=np.full((N,1),0.0)#注意类型要是小数类型, 不能是整数
        r=y.reshape((1,N))
        v=0.0

        Q=cvxopt.matrix(Q)
        p=cvxopt.matrix(p)
        A=cvxopt.matrix(A)
        c=cvxopt.matrix(c)
        r=cvxopt.matrix(r)
        v=cvxopt.matrix(v)

        solution=cvxopt.solvers.qp(Q,p,A,c,r,v)
        alpha=np.array(solution['x'])
        w=np.sum(alpha*y.reshape((N,1))*X,axis=0)
        idx=np.where(alpha>self.epsilon)[0][0]#先取index的list再取list的第一个元素
        b=y[idx]-X[idx,:]@w

        self.alpha=alpha
        self.w=w
        self.b=b

        self.SV=X[np.where(self.alpha.flatten()>self.epsilon)]
    
    def decision_fn(self,X):
        if self.transform_fn:
            X=self.transform_fn.fit_transform(X)
        return self.b+X@self.w

    def predict(self,X):
        return np.sign(self.decision_fn(X))
    
    def eval(self,X,y):
        pred=self.predict(X)
        mistake_indices = np.where(pred!=y)[0]
        accuracy = (X.shape[0]-len(mistake_indices))/X.shape[0]
        return accuracy

# X=np.array([[2,2],[-2,-2],[2,-2],[-2,2]],dtype=np.float)
# y=np.array([1]*2+[-1]*2,dtype=np.float)
# fn=PolynomialFeatures(degree=2,include_bias=False)
# model=Dual_SVM(transform_fn=fn)
# model.fit((X,y))
# hat_y=model.predict(X)
# %%[markdown]
### kernel SVM

# %%
class Kernel_SVM():
    def __init__(self,kernel_name,epsilon=1e-7):
        linear_kernel=lambda x1,x2,gamma:x1.T@x2
        square_kernel=lambda x1,x2,gamma: (1+gamma*x1.T@x2)**2
        cubic_kernel=lambda x1,x2,gamma: (1+gamma*x1.T@x2)**3
        quartic_kernel=lambda x1,x2,gamma: (1+gamma*x1.T@x2)**4
        rbf_kernel=lambda x1,x2,gamma: np.exp(-gamma*np.linalg.norm(x1-x2)**2)

        self.epsilon=epsilon
        
        if kernel_name=='square':
            self.kernel_fn=square_kernel
            self.transform_fn=PolynomialFeatures(degree=2,include_bias=False)
        elif kernel_name=='cubic':
            self.kernel_fn=cubic_kernel
            self.transform_fn=PolynomialFeatures(degree=3,include_bias=False)
        elif kernel_name=='quartic':
            self.kernel_fn=quartic_kernel
            self.transform_fn=PolynomialFeatures(degree=4,include_bias=False)
        elif kernel_name=='rbf':
            self.kernel_fn=rbf_kernel
        else :
            self.kernel_fn=linear_kernel
            self.transform_fn=PolynomialFeatures(degree=1,include_bias=False)
        
        self.kernel_name=kernel_name

    def fit(self,data,gamma=1):
        X=data[0]#注意此处的X是不增广1不升维的X
        y=data[1]
        N,d=X.shape
        self.X=X
        self.y=y.reshape((N,1))
        self.gamma=gamma

        Q=np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Q[i,j]=y[i]*y[j]*self.kernel_fn(X[i,:].T,X[j,:].T,gamma)

        Q=np.array(Q,dtype=np.float)
        p=np.full((N,1),-1,dtype=np.float)
        A=-np.eye(N,dtype=np.float)
        c=np.full((N,1),0.0)#注意类型要是小数类型, 不能是整数
        r=y.reshape((1,N))
        v=0.0

        Q=cvxopt.matrix(Q)
        p=cvxopt.matrix(p)
        A=cvxopt.matrix(A)
        c=cvxopt.matrix(c)
        r=cvxopt.matrix(r)
        v=cvxopt.matrix(v)

        solution=cvxopt.solvers.qp(Q,p,A,c,r,v)
        self.alpha=np.array(solution['x'])
        # w=np.sum(alpha*y.reshape((N,1))*transform_fn.fit_transform(X),axis=0)
        idx=np.where(self.alpha>self.epsilon)[0][0]#先取index的list再取list的第一个元素
        # b=y[idx]-transform_fn.fit_transform(X[idx,:].reshape((1,-1)))@w
        # pred=np.sign(transform_fn.fit_transform(X)@w+b)

        if self.kernel_name!='rbf':
            self.b=y[idx]-(self.alpha*self.y).T@self.kernel_fn(self.X.T,self.X[idx,:],self.gamma) # (1,n)@ # (n,d).T.T@(d,)
        else:
            self.b=y[idx]-(self.alpha*self.y).T@[self.kernel_fn(self.X[n,:],self.X[idx,:],self.gamma) for n in range(self.X.shape[0])]#(1,N)@ (N,)

        self.SV=self.X[np.where(self.alpha.flatten()>self.epsilon)]

        # print('alpha= \n ',solution['x'])
        # print('w= ',w)
        # print('b= ',b)
        # return b,w,transform_fn

    def predict(self,X):
        return np.sign(self.decision_fn(X))

    def decision_fn(self,X):
        if self.kernel_name!='rbf':
            s=self.kernel_fn(X.T,self.X.T,self.gamma)@(self.alpha*self.y)+self.b #(m,d).T.T@(d,n)=(m,n) #(m,n)@(n,1)
        else:
            s=np.zeros((X.shape[0],1))
            for m in range(X.shape[0]):
                tmp=np.zeros(self.X.shape[0]) #(N,)
                for n in range(self.X.shape[0]):
                    tmp[n]=self.kernel_fn(self.X[n,:],X[m,:],self.gamma)
                s[m]=(self.alpha*self.y).T@tmp + self.b # (N,1).T @(N,)
        # return np.sign(s)
        return s
    
    def eval(self,X,y):
        pred=self.predict(X).flatten()#(N,1)与(N,)向量不能直接比较是否对应元素相等
        mistake_indices = np.where(pred!=y)[0]
        accuracy = (X.shape[0]-len(mistake_indices))/X.shape[0]
        return accuracy


# X=np.array([[1,1],[2,2],[2,0],[0,0],[1,0],[0,1]])
# y=np.array([1]*3+[-1]*3,dtype=np.float).reshape((X.shape[0],1))
# model=Kernel_SVM('rbf')
# model.fit((X,y),gamma=0.1)

# %%[markdown]
## T2
# %%

def plot_svm_res(data,model,axes,plot_sv=True):
    X=data[0]
    y=data[1]
    def plot_predictions(model,axes):
        x0s = np.linspace(axes[0], axes[1], 30)
        x1s = np.linspace(axes[2], axes[3], 30)
        x0, x1 = np.meshgrid(x0s, x1s)
        X_ = np.c_[x0.ravel(), x1.ravel()]
        pred=model.decision_fn(X_)
        y_pred = pred.reshape(x0.shape)
        # ax=plt.contourf(x0, x1, np.sign(y_pred), cmap=plt.cm.brg, alpha=0.2)
        cs=plt.contour(x0,x1,y_pred,linewidths=2,levels=[-1,0,1],alpha=0.6)
        # plt.colorbar()
        plt.clabel(cs)
    plt.scatter(X[(y==1).flatten(),0],X[(y==1).flatten(),1],marker='.',color='green')
    plt.scatter(X[(y==-1).flatten(),0],X[(y==-1).flatten(),1],marker='*',color='blue')
    if plot_sv:
        plt.scatter(model.SV[:,0],model.SV[:,1],marker='o',color='pink',alpha=0.6)
    plot_predictions(model,axes)

data=data_generator([-5,0],np.eye(2),[0,5],np.eye(2),400,seed=SEED)
X_train,X_test,y_train,y_test=train_test_split(data[0],data[1],train_size=0.8,test_size=0.2)
train_data=(X_train,y_train)
test_data=(X_test,y_test)
axes=[-8, 4, -4, 8]
# %%

# fn=PolynomialFeatures(degree=2,include_bias=False)
def algorithm(train_data,test_data,axes):
    # primal svm
    model = Primal_SVM()
    model.fit(train_data)

    plt.figure()
    plot_svm_res(train_data,model,axes,plot_sv=False)
    plt.title('Primal SVM train')

    plt.figure()
    plot_svm_res(test_data,model,axes,plot_sv=False)
    plt.title('Primal SVM test')

    print('primal svm train accuracy:',model.eval(test_data[0],test_data[1]))
    print('primal svm test accuracy:',model.eval(test_data[0],test_data[1]))


    # dual svm
    model = Dual_SVM()
    model.fit(train_data)

    plt.figure()
    plot_svm_res(train_data,model,axes)
    plt.title('Dual SVM train')

    plt.figure()
    plot_svm_res(test_data,model,axes,plot_sv=False)
    plt.title('Dual SVM test')

    print('Dual SVM train accuracy:',model.eval(test_data[0],test_data[1]))
    print('Dual SVM test accuracy:',model.eval(test_data[0],test_data[1]))


    # kernel svm
    # quartic polynomial feature 
    model = Kernel_SVM('quartic')
    model.fit(train_data)

    plt.figure()
    plot_svm_res(train_data,model,axes)
    plt.title('quartic kernel SVM train')

    plt.figure()
    plot_svm_res(test_data,model,axes,plot_sv=False)
    plt.title('quartic kernel SVM test')

    print('quartic kernel SVM train accuracy:',model.eval(test_data[0],test_data[1]))
    print('quartic kernel SVM test accuracy:',model.eval(test_data[0],test_data[1]))


    #kernel svm
    # rbf kernel
    model=Kernel_SVM('rbf')
    model.fit(data,gamma=0.1)

    plt.figure()
    plot_svm_res(train_data,model,axes)
    plt.title('rbf kernel SVM train')

    plt.figure()
    plot_svm_res(test_data,model,axes,plot_sv=False)
    plt.title('rbf kernel SVM test')

    print('rbf kernel SVM train accuracy:',model.eval(test_data[0],test_data[1]))
    print('rbf kernel SVM test accuracy:',model.eval(test_data[0],test_data[1]))

    
algorithm(train_data,test_data,axes)

# %%[markdown]
# 从结果图可以看出, 存在边界上的点不是支撑向量的情况, 即对应的alpha=0
## T3

# %%
data=data_generator([3,0],np.eye(2),[0,3],np.eye(2),400,seed=SEED)
X_train,X_test,y_train,y_test=train_test_split(data[0],data[1],train_size=0.8,test_size=0.2)
train_data=(X_train,y_train)
test_data=(X_test,y_test)
algorithm(train_data, test_data, axes)

# %%[markdown]
# 由结果可见, 当数据难以线性区分时, 除了rbf核函数SVM外, 其他几种SVM的结果并不理想

## T4
### 尝试调节核函数中gamma的值
# %%
#kernel svm
# rbf kernel
model=Kernel_SVM('rbf')
model.fit(data,gamma=0.01)

plt.figure()
plot_svm_res(train_data,model,axes)
plt.title('rbf kernel SVM train gamma=0.01')

plt.figure()
plot_svm_res(test_data,model,axes,plot_sv=False)
plt.title('rbf kernel SVM test gamma=0.01')

print('rbf kernel SVM train accuracy:',model.eval(test_data[0],test_data[1]))
print('rbf kernel SVM test accuracy:',model.eval(test_data[0],test_data[1]))

model=Kernel_SVM('rbf')
model.fit(data,gamma=1)

plt.figure()
plot_svm_res(train_data,model,axes)
plt.title('rbf kernel SVM train gamma=1')

plt.figure()
plot_svm_res(test_data,model,axes,plot_sv=False)
plt.title('rbf kernel SVM test gamma=1')

print('rbf kernel SVM train accuracy:',model.eval(test_data[0],test_data[1]))
print('rbf kernel SVM test accuracy:',model.eval(test_data[0],test_data[1]))

# %%[markdown]
# 可见随着gamma的变小, 分类面也会变得更加平滑

### 尝试调节数据量的大小

# %%
data=data_generator([3,0],np.eye(2),[0,3],np.eye(2),50,seed=SEED)
X_train,X_test,y_train,y_test=train_test_split(data[0],data[1],train_size=0.8,test_size=0.2)
train_data=(X_train,y_train)
test_data=(X_test,y_test)
algorithm(train_data, test_data, axes)

# %%[markdown]
# 可见随着样本数量的减少, 数据变得更加线性可分后, 分类效果也有所好转, 同时rbf核SVM与四次核SVM效果均较为理想

## T5
# %%

## 仅仅使用沿海城市

x1=[[119.28,26.08],#福州
[121.31,25.03],#台北
[121.47,31.23],#上海
[118.06,24.27],#厦门
[121.46,39.04],#大连
[122.10,37.50],#威海
[124.23,40.07]]#丹东

x2=[[129.87,32.75],#长崎
[130.33,31.36],#鹿儿岛
[131.42,31.91],#宫崎
[130.24,33.35],#福冈
[133.33,15.43],#鸟取
[138.38,34.98],#静冈
[140.47,36.37]]#水户  
X1=np.vstack((x1,x2))
y1=np.array([1.0]*7+[-1.0]*7)
data1=(X1,y1)

## 钓鱼岛坐标
x=np.array([123.28,25.45]).reshape((1,2))

axes=[115,145,15,40]

model=Kernel_SVM('square')
model.fit(data1)
plot_svm_res(data1,model,axes)
plt.scatter(x[0,0],x[0,1],color='red',marker='X')
label2desc={1:'China',-1:'Japan'}
print('the predict label is ',label2desc[int(model.predict(x))])
# %%[markdown]
# 从预测结果可见, 钓鱼岛是属于中国的
### 增加内陆城市
# %%

# 添加内陆城市

xp1=[[119.28,26.08],#福州
[121.31,25.03],#台北
[121.47,31.23],#上海
[118.06,24.27],#厦门
[113.53,29.58],#武汉
[104.06,30.67],#成都
[116.25,39.54],#北京
[121.46,39.04],#大连
[122.10,37.50],#威海
[124.23,40.07]]#丹东

xp2=[[129.87,32.75],#长崎
[130.33,31.36],#鹿儿岛
[131.42,31.91],#宫崎
[130.24,33.35],#福冈
[136.54,35.10],#名古屋
[132.27,34.24],#广岛
[139.46,35.42],#东京
[133.33,15.43],#鸟取
[138.38,34.98],#静冈
[140.47,36.37]]#水户
X2=np.vstack((xp1,xp2))
y2=np.array([1.0]*10+[-1.0]*10)
data2=(X2,y2)
model=Kernel_SVM('square')
model.fit(data2)
plot_svm_res(data2,model,axes)
plt.scatter(x[0,0],x[0,1],color='red',marker='X')
label2desc={1:'China',-1:'Japan'}
print('the predict label is ',label2desc[int(model.predict(x))])

# %%[markdown]
# 从图中标出的支撑向量可以看出, 增加的内陆城市对于预测并没有造成影响, 支撑向量未发生改变, 而且, 增加内陆城市并不影响分类结果, 钓鱼岛依旧是中国的
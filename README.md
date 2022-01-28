# PatternRecognition
PatternRecongnition homework in HUSTaia 2021 Autumn 

My notebook link: [notebook]( https://www.yuque.com/books/share/140d2549-0462-497d-84bc-da3904529f30)

觉得有用的话, 点个星星呗 😁嘻嘻嘻



| 编号    |   内容     |
| ------ | -------------------------- |
| L2     | **感知器（PLA与Pocket)**      |
| L3     | **线性回归（各类优化器）** |
| L4     | **Fisher线性判别**         |
| L5     | **Logistics回归**          |
| L7~8   | **SVM**                    |
| L9     | **Softmax**                |
| L10~11 | **神经网络与CNN**          |



任务具体要求：

- L2：**感知器（PLA与Pocket)**
  - 分别编写**PLA**算法和**Pocket**算法
  - 产生两个都具有200个二维向量的数据集$𝓧_1$和$𝓧_2$。数据集$𝓧_1$的样本来自均值向量$𝒎_1=[−5,0]^𝑇$、协方差矩阵$𝒔_1=𝑰$的正态分布，属于“+1”类，数据集$𝓧_2$的样本来自均值向量$𝒎_2=[0,5]^𝑇$、协方差矩阵$𝒔_2=𝑰$的正态分布，属于“-1”类，其中$I$是一个2*2的单位矩阵。产生的数据中80%用于训练，20%用于测试。
    - 在上述数据集上分别运用PLA算法和Pocket算法，利用产生的训练样本集得到分类面，算法中用到的各类超参数自定。
    - 分别在训练集和测试集上统计分类正确率
    - 分别统计两个算法的运行时间
    - 画出数据集和分类面
  - 重复上面的内容，但数据集$𝓧_1$和$𝓧_2$的均值向量分别改为$𝒎_1=[1,0]^𝑇$和$𝒎_2=[0,1]^𝑇$，其他不变。

- L3：**线性回归（各类优化器）**

  - 分别编写一个用**广义逆**和**梯度下降法**来求最小误差平方和最佳解的算法
  - 产生两个都具有200个二维向量的数据集$𝓧_1$和$𝓧_2$。数据集$𝓧_1$的样本来自均值向量$𝒎_1=[−5,0]^𝑇$、协方差矩阵$𝒔_1=𝑰$的正态分布，属于“+1”类，数据集$𝓧_2$的样本来自均值向量$𝒎_2=[0,5]^𝑇$、协方差矩阵$𝒔_2=𝑰$的正态分布，属于“-1”类，其中$I$是一个2*2的单位矩阵。产生的数据中80%用于训练，20%用于测试。
    - 在上述数据集上分别使用第1题的两个算法，利用产生的训练样本集得到分类面，算法中用到的各类超参数自定。
    - 分别在训练集和测试集上统计分类正确率
    - 画出数据集和分类面
    - 画出损失函数随epoch增加的变化曲线
  - 重复上面的内容，但数据集$𝓧_1$和$𝓧_2$的均值向量分别改为$𝒎_1=[1,0]^𝑇$和$𝒎_2=[0,1]^𝑇$，其他不变。
  - 改变算法中的各类超参数、样本数量、样本分布等，对于梯度下降法还要改变不同的学习率以及不同的batch size和不同epoch次数，讨论实验结果。
  - 单变量函数为𝑓(𝑥)=𝑥∗cos (0.25𝜋∗𝑥)，分别用梯度下降法、随机梯度下降法、**Adagrad**、**RMSProp**、动量法（**Momentum**）和**Adam**共6种方法，编写程序画图呈现𝑥𝑥从初始值为-4、迭代10次时𝑥及𝑓(𝑥)的每次变化情况，这里对所有算法学习率（或初始学习率）均为0.4，为防止分母为0时给的最小量为𝜀=1e-6，RMSProp算法的𝛼=0.9，动量法的𝜆=0.9，Adam的beta1=0.9，beta2=0.999，观察不同算法的变化情况体会各自的差异。如果迭代50次，并将Adam的beta1改成0.99，其他参数不变，观察不同算法的变化结果。尝试调整上述算法的各种参数，体会上述不同方法的特点。

- L4：**Fisher线性判别**

  - 编程实现Fisher线性判别算法
  - 产生两个都具有200个二维向量的数据集$𝓧_1$和$𝓧_2$。数据集$𝓧_1$的样本来自均值向量$𝒎_1=[−5,0]^𝑇$、协方差矩阵$𝒔_1=𝑰$的正态分布，属于“+1”类，数据集$𝓧_2$的样本来自均值向量$𝒎_2=[0,5]^𝑇$、协方差矩阵$𝒔_2=𝑰$的正态分布，属于“-1”类，其中$I$是一个2*2的单位矩阵。产生的数据中80%用于训练，20%用于测试。
    - 在上述数据集上运用Fisher线性判别算法，在产生的训练样本集上得到最佳投影向量，并计算出分类阈值。
    - 在训练集和测试集上分别统计分类正确率。
    - 画出数据集、最佳投影向量和分类阈值。

- L5：**Logistics回归**

  - 编程实现Logistic regression算法。
  - 产生两个都具有200个二维向量的数据集$𝓧_1$和$𝓧_2$。数据集$𝓧_1$的样本来自均值向量$𝒎_1=[−5,0]^𝑇$、协方差矩阵$𝒔_1=𝑰$的正态分布，属于“+1”类，数据集$𝓧_2$的样本来自均值向量$𝒎_2=[0,5]^𝑇$、协方差矩阵$𝒔_2=𝑰$的正态分布，属于“-1”类，其中$I$是一个2*2的单位矩阵。产生的数据中80%用于训练，20%用于测试。
    - 在训练集上利用Logistic regression算法得到分类面。
    - 利用得到的分类面对测试集样本进行分类，并给出每个样本属于该类别的概率值

- L7~8：**SVM**

  - 利用二次规划函数，分别编程实现原问题求解的支撑向量机算法（**Primal-SVM**）、对偶的支撑向量机算法（**Dual-SVM**）、和核函数的支撑向量机算法（**Kernel-SVM**）。
  - 产生两个都具有200个二维向量的数据集$𝓧_1$和$𝓧_2$。数据集$𝓧_1$的样本来自均值向量$𝒎_1=[−5,0]^𝑇$、协方差矩阵$𝒔_1=𝑰$的正态分布，属于“+1”类，数据集$𝓧_2$的样本来自均值向量$𝒎_2=[0,5]^𝑇$、协方差矩阵$𝒔_2=𝑰$的正态分布，属于“-1”类，其中$I$是一个2*2的单位矩阵。产生的数据中80%用于训练，20%用于测试。
    - 在上述数据集上分别运用Primal-SVM、Dual-SVM和Kernel-SVM算法，利用产生的训练样本集得到分类面，其中，Kernel-SVM中的核函数分别采用四次多项式和高斯核函数，算法中用到的各类超参数自定。
    - 分别在训练集和测试集上统计分类正确率
    - 对于Dual-SVM和Kernel-SVM算法，指出哪些样本是支撑向量
    - 画出数据集和分类面、间隔面，并标注出哪些样本是支撑向量，观察是否有边界上的向量不是支撑向量的现象。
    - 重复上面的内容，但数据集$𝓧_1$和$𝓧_2$的均值向量分别改为$𝒎_1=[3,0]^𝑇$和$𝒎_2=[0,3]^𝑇$，其他不变。
    - 改变算法中的超参数、样本数量、样本分布等，讨论实验结果。
  - 训练集: 中国与日本的沿海城市的经纬度坐标向量，中国标签为+1, 日本为标签为-1.
    测试集: 钓鱼岛的经纬度坐标向量，用支撑向量机设计分类器。
    - 判断钓鱼岛属于哪一类；
    - 增加几个非海边城市的经纬度坐标进行训练，判断这些城市是否影响分类结果，是否为支撑向量。

- L9：**Softmax**

  - 给定IRIS数据集，该数据集有三类目标，每个类别有50个样本，每个样本有四维特征。实验时每个类别随机选30个样本进行训练，另外20个样本用于测试。
    - 以感知器算法为基础分类算法，编写一个**OVO多类分类器**算法，对上述数据集进行实验，分析结果。
    - 编写Softmax算法实现多类别分类，对上述数据集进行实验，分析结果。
  - 给定MNIST数据集，该数据集每个样本为28*28大小的灰度图像，有0到9共10个类别的手写体数字，其中训练样本60000，测试样本10000
    - 编写**Softmax**算法对该数据集实现分类，权向量初始值由均值为0、标准差为0.01的正态分布产生的随机数得到，统计此时测试集的分类精度（正确分类的样本数/总样本数）。训练时的batch size为256，一共训练10遍epoch
    - 画出训练时的损失函数、训练集上的分类精度和测试集上的分类精度随epoch增加的变化曲线。训练完成后，在测试集上随机抽取10个样本，观察分类结果。

- L10~11：**神经网络与CNN**

  - IRIS数据集有三类目标，每个类别有50个样本，每个样本有四维特征。自行设计神经网络实现对这三个目标的识别，实验时每个类别随机选30个样本进行训练，另外20个样本用于测试。希望能通过设计不同的隐含层数、每层的节点数、不同的学习率、不同的激活函数等对实验结果进行讨论。

  - 2，LeNet网络结构如下：

    - 第1层卷积层Conv-1： 6个5*5*1大小的滤波器， stride=1，padding=2，接Sigmoid做激活函数；接下来是池化层AvePool-1，它以2*2、stride=2做Average Pooling操作；第2层卷积层 Conv-2： 16个5*\*5\*6大小的滤波器， stride=1，padding=0，接Sigmoid做激活函数；再接一个池化层 AvePool--2，它以2*2、stride=2做Average Pooling操作；对AvePool--2层输出做了Flatten操作后，与120个神经元做全连接，构成FC-1，Sigmoid做激活函数；再与84个神经元做全连接，构成FC-2，Sigmoid做激活函数；再全连接10个神经元输出，用Softmax完成10个类别的分类。
    - 编写上述网络结构的代码，对MNIST数据集实现分类，训练时的batch size为256，一共训练10遍epoch，画出训练时的损失函数、训练集上的分类精度和测试集上的分类精度随epoch增加的变化曲线。训练完成后，在测试集上随机抽取10个样本，观察分类结果。

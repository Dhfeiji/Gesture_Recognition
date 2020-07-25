import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 分类问题

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# 把x和y都变成Variable的形式---> 因为神经网络只能输入Variable
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):  # 继承torch.nn.Module模块：Net的主模块
    # 官方步骤 (定义网络层)
    def __init__(self, n_feature, n_hidden, n_output):  # 存放搭建神经层所需的一些信息
        super(Net, self).__init__()   # 搭建网络图之前要继承Net中的Module模块
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 神经网络的第一层hidden层(输入数据的个数，本隐藏层神经节点个数)
        self.out = torch.nn.Linear(n_hidden, n_output)  # 第二个神经层：预测神经层predict层
        # torch.nn.Linear中Linear是全连接，nn是神经网络

    # 网络的搭建过程
    def forward(self, x):  # 神经网络前向传播的一个过程。 x:网络的输入信息
        x = F.relu(self.hidden(x))  # x经过hidden层输出后，再进行激活函数操作
        x = self.out(x) # 上一层的输出放到predict层来进行预测，输出预测结果x
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)   # 参数：输入1个值， 隐藏层有10个神经元， 输出为1个值
print(net)  # 查看神经网络的层结构

# 定义优化器：优化神经网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 使用SGD优化器来优化net神经网络的所有参数, 学习率:0.2

# 定义损失函数：计算误差
loss_func = torch.nn.CrossEntropyLoss()

# 可视化
plt.ion()  # 设置实时打印的过程

# 开始训练神经网络
for t in range(100):  # 训练的步数：100
    # 预测
    out = net(x)  # 查看每一步的预测结果

    # 计算预测值与真实值的对比：计算误差损失
    loss = loss_func(out, y)

    # 网络优化的步骤
    # 梯度归零
    optimizer.zero_grad()  # SGD优化神经网络的所有参数：先把所有参数的梯度都初始化为0
    # 反向传播：计算节点的梯度
    loss.backward()
    # optimizer以学习率0.5来优化每一步的梯度
    optimizer.step()

    # 可视化打印
    if t % 2 == 0:  # 每学习5步打印一次
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

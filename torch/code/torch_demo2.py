import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # torch.unsqueeze将一维数据变为二维
y = x.pow(2) + 0.2*torch.rand(x.size())  # x.pow(2): x的二次方;后面是一些噪音(这个y就是x对应的真实值)

# 把x和y都变成Variable的形式---> 因为神经网络只能输入Variable
x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())  # plt.scatter打印散点图（先转换为numpy形式数据）
plt.show()

class Net(torch.nn.Module):  # 继承torch.nn.Module模块：Net的主模块
    # 官方步骤 (定义网络层)
    def __init__(self, n_feature, n_hidden, n_output):  # 存放搭建神经层所需的一些信息
        super(Net, self).__init__()   # 搭建网络图之前要继承Net中的Module模块
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 神经网络的第一层hidden层(输入数据的个数，本隐藏层神经节点个数)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 第二个神经层：预测神经层predict层
        # torch.nn.Linear中Linear是全连接，nn是神经网络

    # 网络的搭建过程
    def forward(self, x):  # 神经网络前向传播的一个过程。 x:网络的输入信息
        x = F.relu(self.hidden(x))  # x经过hidden层输出后，再进行激活函数操作
        x = self.predict(x) # 上一层的输出放到predict层来进行预测，输出预测结果x
        return x

net = Net(1, 10, 1)   # 参数：输入1个值， 隐藏层有10个神经元， 输出为1个值
print(net)  # 查看神经网络的层结构

# 可视化
plt.ion()  # 设置实时打印的过程
plt.show()

# 定义优化器：优化神经网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 使用SGD优化器来优化net神经网络的所有参数, 学习率:0.2

# 定义损失函数：计算误差
loss_func = torch.nn.MSELoss()  # MSE均方差，用均方差来处理回归问题(如果是分类问题，使用交叉熵损失)

# 开始训练神经网络
for t in range(100):  # 训练的步数：100
    # 预测
    prediction = net(x)  # 查看每一步的预测结果

    # 计算预测值与真实值的对比：计算误差损失
    loss = loss_func(prediction, y)

    # 网络优化的步骤
    # 梯度归零
    optimizer.zero_grad()  # SGD优化神经网络的所有参数：先把所有参数的梯度都初始化为0
    # 反向传播：计算节点的梯度
    loss.backward()
    # optimizer以学习率0.5来优化每一步的梯度
    optimizer.step()

    # 可视化打印
    if t % 5 == 0:  # 每学习5步打印一次
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())  # plt.scatter散点图（原始数据）
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 神经网络预测的曲线(学习程度)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})# 打印学习过程中的误差
        plt.pause(0.1)

plt.ioff()
plt.show()

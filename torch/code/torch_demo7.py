import torch
import torch.utils.data as Data
import torch.nn.functional as F
# from torch.autograd import Variable
import matplotlib.pyplot as plt

"""
    四种优化器的对神经网络的优化效果
"""

# 超参数
LR = 0.01
BATCH_SIZE = 32  # 小批量数据
EPOCH = 12

# 数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)   # 二维数据(回归数据)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# 绘制数据图
# plt.scatter(x.numpy(), y.numpy())  # scatter散点图
# plt.show()

# 用mini-batch_size数据训练   --- loader：训练数据
torch_dataset = Data.TensorDataset(x, y) # x是data_tensor   y是target_tensor
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # num_workers=2线程为2

# 搭建神经网络结构
class Net(torch.nn.Module):
    # 官方步骤 (定义网络层)
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    # 网络的搭建过程
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':
    # 命名四个神经网络(用一个list存储)--使用不同的优化器来进行优化编译同一个网络(效果比较)
    net_SGD      = Net()
    net_Momentum = Net()
    net_RMSprop  = Net()
    net_Adam     = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]   # 之后用for循环来进行依次训练优化

    # 建立四个不同的优化器(用一个list存储)
    opt_SGD      = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop  = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam     = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers    = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_history = [[],[],[],[]]   # list形式：记录误差损失变化曲线

    for epoch in range(EPOCH):
        print(epoch)
        for step, (batch_x, batch_y) in enumerate(loader):
            # b_x = Variable(batch_x)  # 训练之前需要将数据变为Variable形式。因为DataLoader传进去的x和y都是tensor形式的
            # b_y = Variable(batch_y)  # 现在不需要将tensor数据用Variable包装了，因为tensor已经包含了Variable功能！！
            for net, opt, l_history in zip(nets, optimizers, losses_history):  # zip捆在一起，然后一组一组的提取(net, opt, l_history)
                # training
                output = net(batch_x)
                loss = loss_func(output, batch_y)   # 计算每个网络的损失
                opt.zero_grad()  # 梯度初始化为0：clean gradients for next train
                loss.backward()  # 反向传播，计算梯度
                opt.step()       # 应用梯度  apply gradients
                l_history.append(loss.item())  # loss record

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_history in enumerate(losses_history):
        plt.plot(l_history, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()
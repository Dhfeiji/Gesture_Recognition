import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)   # 使用np.linspace定义x：范围是(-5,5);个数是200.  x data (tensor), shape(100,1)
x = Variable(x)  # 把数据套在Variable的篮子中(就变成Variable了)
# 画图的时候torch的数据无法被matplotlib识别，所以画图前转换为numpy的数据形式
x_np = x.data.numpy()   # tensor是存放在data里面的，所以先.data

y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_relu = F.relu(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()


plt.figure(1, figsize=(8, 6))    # figure定义一个窗口，编号为1，大小为(8, 6)

plt.subplot(221)   # 221: 表示将整个图像窗口分为2行2列, 当前位置为1
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')  # 使用plt.plot画(x_np, y_sigmoid)曲线，颜色是red，标签是'sigmoid'
plt.xlim(-4, 4)
plt.ylim((-0.2, 1.2))
plt.xlabel('I am x')   # 设置横纵坐标轴名称
plt.ylabel('I am y')
plt.legend(loc='best')  # legend 图例就是为了帮我们展示出每个数据对应的图像名称。’best’表示自动分配最佳位置
# plt.legend(loc='upper right')  表示图例将添加在图中的右上角

plt.subplot(222)
plt.plot(x_np, y_tanh, c='red', linewidth=1.0, linestyle='--', label='tanh')  # 曲线的宽度(linewidth)为1.0.曲线的类型(linestyle)为虚线
plt.xlim(-4, 4)  # 使用plt.xlim设置x坐标轴范围: (-4, 4)
plt.ylim((-1.2, 1.2))  # 使用plt.ylim设置x坐标轴范围：(-1.2, 1.2)
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.xlim(-4, 4)
plt.ylim((-1, 5))
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.xlim(-4, 4)
plt.ylim((-0.2, 6))
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.legend(loc='best')
plt.show()
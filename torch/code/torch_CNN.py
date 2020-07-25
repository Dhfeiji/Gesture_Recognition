import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision   # 数据库，里面存放了各种经典模型和数据集(如VGGNet、Mnist)
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001         # learning rate
DOWNLOAD_MNIST = False  # 如果数据集已经下好了，就设为False

train_data = torchvision.datasets.MNIST(  # 这行代码就是去网站下载mnist数据集(train+test一起下载的)
    root = './mnist',   # 保存路径
    train = True,   # True 的话下载的是训练数据，False则下载的是测试数据
    transform = torchvision.transforms.ToTensor(),  # 转换为tensor的格式:(0-255) -> (0,1)
    download = DOWNLOAD_MNIST
)
# plot one example
# print(train_data.train_data.size())  # (60000, 28, 28)
# print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
# 把train_data变成train_loader的形式
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)  # 从目录提取test_data
# unsqueeze是将提取到的test_data加上batch_size维度
# 转换数据类型，再归一化
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.0
test_y = test_data.test_labels

"""定义CNN网络"""
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Sequential(  # 第一部分：包含一个卷积一个池化.    input:|->(1, 28, 28)
            nn.Conv2d(               # 第一个卷积层      |->(16, 28, 28)
                in_channels=1,       # 输入数据的通道数
                out_channels=16,     # 输出数据的通道数--即filter个数
                kernel_size=5,       # 5x5的filter
                stride=1,            # 步长为1
                padding=2            # padding = (kernel_size - stride)/2
            ),
            nn.ReLU(),          # |->(16, 28, 28)
            nn.MaxPool2d(kernel_size=2)  # 2x2的filter    |->(16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # 第二部分：包含一个卷积一个池化.    input:|->(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),   # |->(32, 14, 14)
            nn.ReLU(),                    # |->(32, 14, 14)
            nn.MaxPool2d(2)               # |->(32, 7, 7)
        )
        self.fuc = nn.Linear(32*7*7, 128)   # 第一层全连接: (128)
        self.out  = nn.Linear(128, 10)       # 第二层全连接: (128-->10分类)

    def forward(self, x):
        x = self.conv1(x)           # 第一部分操作
        x = self.conv2(x)           # 第二部分操作   (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # 类似Flatten：平展开-->  (batch, 32, 7, 7) --> (batch, 32*7*7)

        x = self.fuc(x)             # 第一层全连接
        output = self.out(x)        # 第二层全连接
        return output

if __name__ == '__main__':
    cnn = CNN_Net()
    print(cnn)   # net architecture
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func =nn.CrossEntropyLoss()   # CrossEntropyLoss自带softmax

    # training and testing
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)
            loss = loss_func(output, batch_y)  # 计算每个网络的损失
            optimizer.zero_grad()  # 梯度初始化为0：clean gradients for next train
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度  apply gradients

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    # print 10 predictions from test data
    test_output = cnn(test_x[:10])   # 放前十张图片进去，预测输出
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
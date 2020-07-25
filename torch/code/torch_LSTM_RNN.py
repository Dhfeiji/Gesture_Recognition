import torch
from torch import nn
import torch.utils.data as Data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64    # 披训练的数量
TIME_STEP = 28     # run time step / image height  --> 每一步读取图片的一行(共28行)--28个时间点
INPUT_SIZE = 28    # run input size / image width  --> 每一行有28个像素点(每一个时间点给RNN28个像素点)
LR = 0.01          # learning rate
DOWNLOAD_MNIST = False   # 数据集已经下好了

train_data = datasets.MNIST(root='../mnist',train=True, transform=transforms.ToTensor(),download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='../mnist', train=False)
test_x = test_data.test_data.type(torch.FloatTensor)/255.0
test_y = test_data.test_labels

"""定义RNN网络"""
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(       # 使用LSTM的RNN形式
            input_size=28,
            hidden_size=64,
            num_layers=2,     # 2表示有两层hidden_layer
            batch_first=True,         # (batch_size, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)   # 全连接层： 64-->10

    def forward(self, x):
        # LSTM生成的hidden state：一个分线程的，一个主线程的
        r_out, (h_n, h_c) = self.rnn(x, None)  # x: (batch, time_step, input_size)。第一个hidden state没有，所以为None
        out = self.out(r_out[:, -1, :])  # x的格式，中间是time_step，所以此处-1表示选取最后一个时刻的r_out
        return out

if __name__ == '__main__':
    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
            b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

            output = rnn(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                test_output = rnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    # print 10 predictions from test data
    test_output = rnn(test_x[:10].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')
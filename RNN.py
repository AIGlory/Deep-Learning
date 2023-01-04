import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device('cuda:0')

EPOCHS = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
# 学习率设置为 0.01
LEARN_RATE = 0.01
#DATA_PATH_ROOT+
train_data = torchvision.datasets.MNIST(
    root='data',
    train=True,
    transform=transforms.ToTensor(),  # 将下载的文件转换成pytorch认识的tensor类型，且将图片的数值大小从（0-255）归一化到（0-1）
)
test_data = torchvision.datasets.MNIST(
    root= 'data',
    train=False,
    transform=transforms.ToTensor()  # 将下载的文件转换成pytorch认识的tensor类型，且将图片的数值大小从（0-255）归一化到（0-1）
)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # 这里使用的是 torch 的网络
        self.model = nn.RNN(
        #self.model = nn.LSTM(
        #self.model = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, _ = self.model(x)
        out = self.out(r_out[:, -1, :])
        return out

# 实例化RNN，并将模型放在 GPU 上训练
model = model().to(device)
# 使用交叉熵损失，同样，将损失函数放在 GPU 上
loss_fn = nn.CrossEntropyLoss().to(device)
# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
for i in range(EPOCHS):
    for step, data in enumerate(train_loader):
        # 分别得到训练数据的x和y的取值
        x, y = data
        x, y = x.to(device), y.to(device)
        x = x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        output = model(x).to(device)  # 调用模型预测
        loss = loss_fn(output, y.long())  # 计算损失值
        optimizer.zero_grad()  # 每一次循环之前，将梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

sum = 0
# test：
for i, data in enumerate(test_loader):
    x, y = data
    x, y = x.to(device), y.to(device)
    x = x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
    # 得到模型预测输出，10个输出，即该图片为每个数字的概率
    res = model(x)
    # 最大概率的就为预测值
    r = torch.argmax(res)
    l = y.item()
    sum += 1 if r == l else 0
    print(f'test({i})     RNN:{r} -- label:{l}')
    # print(f'test({i})     LSTM:{r} -- label:{l}')
    # print(f'test({i})     GRU:{r} -- label:{l}')
print('accuracy：', sum/10000)
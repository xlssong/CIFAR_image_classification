import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torchvision
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from data_utils import load_data
from model import Net
import config
import torch.nn.functional as F


device = config.device
# 超参数
batch_size = config.batch_size
embedding_dim = config.embedding_dim
hidden_dim = config.hidden_dim

num_epochs = 30

def train(model, trainloader):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    writer = SummaryWriter(log_dir='./runs/cifar10_experiment')
    # 训练模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # 完整训练过程：
            # 1.       取数据        inputs, labels
            # 2.        清零梯度        optimizer.zero_grad()
            # 3.        前向传播        outputs = model(inputs)
            # 4.        计算损失        loss = criterion(outputs, labels)
            # 5.        反向传播        loss.backward()
            # 6.        更新参数        optimizer.step()
            # 7.        记录损失和准确率
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 正向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}')
                writer.add_scalar('Training Loss', loss.item(), epoch * len(trainloader) + i)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}, Accuracy: {100 * correct / total:.2f}%")

def eval(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

model = Net().to(device)
trainloader, testloader = load_data()
train(model, trainloader)
eval(model, testloader)
torch.save(model.state_dict(), config.model_name)
print(f"模型已保存为 {config.model_name}")
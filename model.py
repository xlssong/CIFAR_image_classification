# model.py - CNN Model for Image Classification (CIFAR-10)
# Author: xlssong
# Date: 2025-03-08
# Description: This script defines a convolutional neural network (CNN) using PyTorch
# to perform image classification on the CIFAR-10 dataset. The model consists of three
# convolutional layers, max-pooling, and three fully connected layers for classification.

import torch.nn as nn
import torch.nn.functional as F

# 定义 CNN 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1：输入3个通道，输出32个通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：将数据展平，最终输出10个类别
        self.fc1 = nn.Linear(256 * 4 * 4, 128)  # 输入尺寸相应改变
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # CIFAR-10 有 10 个类别
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层 -> 激活函数 -> 池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # 展平 -> 全连接层
        x = x.view(-1, 256 * 4 * 4)
        #x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

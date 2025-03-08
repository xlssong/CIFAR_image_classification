import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import config
from model import Net
import torch.nn.functional as F
import data_utils

device = config.device
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_utils.transform)

def random_inference(model, testset, classes, num_images=5):
    model.eval()
    indices = random.sample(range(len(testset)), num_images)  # 随机抽样
    images = [testset[i][0] for i in indices]
    labels = [testset[i][1] for i in indices]

    inputs = torch.stack(images).to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # # 显示图片和预测结果
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        img = img * 0.225 + 0.45  # 反标准化
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
        ax.axis('off')
    plt.show()

model = Net().to(device)
model.load_state_dict(torch.load(config.model_name))
model.eval()  # 切换到推理模式
# 使用训练好的模型进行推理
random_inference(model, testset, data_utils.classes)

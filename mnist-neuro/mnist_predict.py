import random

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import datasets

from model.CNN import CNN

# 加载MNIST测试数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 加载已训练的模型
model = CNN()
model.load_state_dict(torch.load('./model/mnist_cnn_model.pth'))
model.eval()

# 选择一张测试图像进行预测
image_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[image_index]
image = image.unsqueeze(0)

# 进行预测
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    predicted_label = predicted.item()

# 可视化预测结果
image = image.squeeze().numpy()
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {label}")
plt.axis('off')
plt.show()

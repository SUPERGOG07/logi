import random

import matplotlib.pyplot as plt
from torchvision import datasets

# 加载MNIST数据集
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)

# 可视化前几张图像
num_images = 5
fig, axes = plt.subplots(1, num_images, figsize=(10, 3))

random_indices = random.sample(range(len(mnist_dataset)), num_images)

for i, index in enumerate(random_indices):
    image, label = mnist_dataset[index]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

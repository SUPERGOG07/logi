import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from model import VGG

num_epochs = 10
batch_size = 128
lr = 0.01
momentum = 0.9
weight_decay = 5e-4

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载训练集和测试集
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 选择模型
model = VGG.vgg16_bn()
# model = densenet.DenseNet121()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 训练函数
def train(epoch):
    total_step = len(train_loader)
    model.train()
    train_loss = 0.0

    with tqdm(total=total_step, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({"Loss": train_loss / (i + 1)})
            pbar.update()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    return train_loss / len(train_loader)


# 测试函数
def test(epoch):
    global best_accuracy
    model.eval()
    test_correct = 0
    test_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_samples += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    accuracy = test_correct / test_samples * 100

    if accuracy > best_accuracy:
        print('Saving model...')
        torch.save(model.state_dict(), './pth/cifar10_vgg16.pth')
        # torch.save(model.state_dict(), './pth/cifar10_densenet121.pth')
        best_accuracy = accuracy

    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%, Best Accuracy: {best_accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':

    best_accuracy = 0
    for epoch in range(num_epochs):
        train_loss = train(epoch)
        test_accuracy = test(epoch)

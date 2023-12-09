import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from network import VGG,densenet

batch_size = 128

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 选择模型
# network = VGG.vgg16_bn()
model = densenet.DenseNet121()

# network.load_state_dict(torch.load('./pth/cifar10_vgg16.pth'))
model.load_state_dict(torch.load('./pth/cifar10_densenet121.pth'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 测试函数
def test():
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

    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    test()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset_path = "/Users/Mukhil/Desktop/Sonoma_Hacks/Garbage classification/Garbage classification"




class AttentionGate(nn.Module):
    def __init__(self, channels):
        super(AttentionGate, self).__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer1 = nn.Linear(channels, channels)
        self.linear_layer2 = nn.Linear(channels, channels)

    def forward(self, x):
        a, b, c, d = x.shape
        pooled = self.globalpool(x).view(a, b)
        weights = F.relu(self.linear_layer1(pooled))
        weights = torch.sigmoid(self.linear_layer2(weights))
        weights = weights.view(a, b, 1, 1)
        return x * weights # Explanation of this line below
# Element multiplication so weights are only multiplied to corresponding elements
# in this case channel information. (a,b) are multiplied ONLY to corresponding weights.
# this means spactial information (c,d) aren't affected and are preserved. (1*x=x) because Element multiplication
# Multiplies elements only with corresponding elements so 1*spactial_info=spactial_info preserved.



class BinaryTrashDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.dataset = ImageFolder(folder_path, transform=transform)
        self.transform = transform
        self.class_to_idx = self.dataset.class_to_idx
        self.trash_indices = {self.class_to_idx[name] for name in ["plastic", "trash"]}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        binary_label = 1 if label in self.trash_indices else 0
        return image, binary_label

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

        self.attn = AttentionGate(256)

        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = self.attn(x)

        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

full_dataset = BinaryTrashDataset(dataset_path, transform=transform)
total_size = len(full_dataset)
train_size = int(0.86 * total_size)
val_size = int(0.06 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

net = model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)

for epoch in range(30):
    net.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/35", ncols=100):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

torch.save(net.state_dict(), "model.pth")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as T
import numpy as np
import urllib.request
import os
from monai.transforms import RandFlipd, RandRotate90d

# Descargar OrganMNIST3D si no existe
url = 'https://zenodo.org/record/6496656/files/organmnist3d.npz'
npz_path = 'organmnist3d.npz'
if not os.path.exists(npz_path):
    print('Descargando OrganMNIST3D...')
    urllib.request.urlretrieve(url, npz_path)

# Cargar datos
npz = np.load(npz_path)
train_images = npz['train_images'].astype(np.float32) / 255.0
train_labels = npz['train_labels'].astype(np.int64).squeeze()
val_images = npz['val_images'].astype(np.float32) / 255.0
val_labels = npz['val_labels'].astype(np.int64).squeeze()

# Dataset personalizado para 3D
class Custom3DDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        img = torch.tensor(img).unsqueeze(0)  # añadir canal C=1
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)

# Augmentations
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
])

val_transform = T.Compose([])

train_dataset = Custom3DDataset(train_images, train_labels, transform=None)  # De momento sin augment
val_dataset = Custom3DDataset(val_images, val_labels, transform=None)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Modelo ResNet3D
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ResNet3D(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.layer1 = BasicBlock3D(1, 32)
        self.layer2 = BasicBlock3D(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0

for epoch in range(300):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total

    print(f"Epoch {epoch+1}, Train loss: {avg_loss:.4f}, Validation accuracy: {acc:.2f}%")

    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_resnet3d_organmnist.pt")
        print(f"✅ Nuevo mejor modelo guardado (acc: {best_acc:.2f}%)")

print("Entrenamiento completado. Modelo guardado en 'best_resnet3d_organmnist.pt'")


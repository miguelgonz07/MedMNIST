import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import urllib.request
import os
from PIL import Image

# Descargar OrganMNISTAxial si no existe
url = 'https://zenodo.org/record/6496656/files/organmnist_axial.npz?download=1'
npz_path = 'organmnist_axial.npz'
if not os.path.exists(npz_path):
    print('Descargando OrganMNISTAxial...')
    urllib.request.urlretrieve(url, npz_path)

# Cargar datos
data = np.load(npz_path)
train_images = data['train_images'].astype(np.float32) / 255.0
train_labels = data['train_labels'].astype(np.int64).squeeze()
val_images = data['val_images'].astype(np.float32) / 255.0
val_labels = data['val_labels'].astype(np.int64).squeeze()

# Dataset personalizado con augmentations
class CustomTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)

# Transformaciones (augmentations)
transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor()
])

transform_val = T.Compose([
    T.ToTensor()
])

# Modelo ResNet2D mejorado
class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)

class ResNet2D(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.layer1 = BasicBlock2D(1, 64)
        self.layer2 = BasicBlock2D(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Solo ejecuta esto si corres directamente el script
if __name__ == "__main__":
    train_dataset = CustomTensorDataset(train_images, train_labels, transform=transform_train)
    val_dataset = CustomTensorDataset(val_images, val_labels, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(500):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                pred_labels = torch.argmax(preds, dim=1)
                correct += (pred_labels == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Train loss: {avg_loss:.4f}, Validation accuracy: {acc:.2f}%")

        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_resnet2d_organmnist.pt")
            print(f"✅ Nuevo mejor modelo guardado (acc: {best_acc:.2f}%)")

    print("Entrenamiento completado. Modelo guardado en 'best_resnet2d_organmnist.pt'")


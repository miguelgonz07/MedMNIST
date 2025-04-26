import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import INFO, PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
#from scripts.unet import UNet
from unet import UNet
from utils import dice_score

# Configuración
info = INFO['pathmnist']
DataClass = PathMNIST
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

# Transformaciones y datos
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = DataClass(split='train', transform=transform, download=True)
val_dataset = DataClass(split='val', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Modelo, pérdida, optimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Entrenamiento
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = (labels > 0).float().unsqueeze(1).to(device)  # binary seg target

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # Validación
    model.eval()
    dices = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = (labels > 0).float().unsqueeze(1).to(device)
            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).float()
            dices.append(dice_score(preds, labels))
    print(f"Validation Dice: {sum(dices)/len(dices):.4f}")

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model2d import ResNet2D

# Cargar datos
data = np.load('organmnist_axial.npz')
test_images = data['test_images'].astype(np.float32) / 255.0
test_labels = data['test_labels'].astype(np.int64).squeeze()

test_x = torch.tensor(test_images).unsqueeze(1)
test_y = torch.tensor(test_labels)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet2D().to(device)
model.load_state_dict(torch.load('best_resnet2d_organmnist.pt'))
model.eval()

# Evaluar
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        pred_labels = torch.argmax(preds, dim=1)
        correct += (pred_labels == y).sum().item()
        total += y.size(0)

acc = 100 * correct / total
print(f"✅ Precisión en test set: {acc:.2f}% ({correct}/{total})")


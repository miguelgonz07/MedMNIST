import torch
import numpy as np
import random
from model3d import ResNet3D

# Cargar datos
data = np.load('organmnist3d.npz')
test_images = data['test_images'].astype(np.float32) / 255.0
test_labels = data['test_labels'].astype(np.int64).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet3D().to(device)
model.load_state_dict(torch.load('best_resnet3d_organmnist.pt'))
model.eval()

# Seleccionar aleatoriamente una imagen
idx = random.randint(0, len(test_images) - 1)
sample = torch.tensor(test_images[idx]).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 28, 28, 28)
true_label = test_labels[idx]

# Inferencia
with torch.no_grad():
    output = model(sample)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"✅ Predicción del modelo: Clase {predicted_label}")
print(f"🎯 Clase verdadera (ground truth): {true_label}")


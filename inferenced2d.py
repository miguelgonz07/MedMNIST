import torch
import numpy as np
import random
from model2d import ResNet2D  # Importa tu modelo 2D

# Cargar los datos de test
data = np.load('organmnist_axial.npz')
test_images = data['test_images'].astype(np.float32) / 255.0
test_labels = data['test_labels'].astype(np.int64).squeeze()

# Seleccionar una imagen aleatoria
idx = random.randint(0, len(test_images) - 1)
image = test_images[idx]
label = test_labels[idx]

# Preparar el tensor
image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet2D().to(device)
model.load_state_dict(torch.load('best_resnet2d_organmnist.pt'))
model.eval()

# Inferencia
image_tensor = image_tensor.to(device)
with torch.no_grad():
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

print(f"✅ Predicción del modelo: Clase {pred_class}")
print(f"✯⃝ Clase verdadera (ground truth): {label}")


import torch

def dice_score(pred, target, epsilon=1e-6):
    # Asume pred y target con dimensiones [B, 1, H, W]
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

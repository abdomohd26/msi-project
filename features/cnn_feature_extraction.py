import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet18.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image_to_feature_cnn(arr):
    """
    CNN features from numpy array: ResNet18 (512 dims).
    """
    # Convert numpy [H,W,3] [0,1] -> PIL -> tensor
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)  # [0-255]
    img = Image.fromarray(arr).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = resnet18(img_t).cpu().numpy().flatten()
    
    return features  # (512,)
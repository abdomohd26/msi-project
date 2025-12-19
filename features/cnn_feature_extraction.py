import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from torchvision.models import (
    efficientnet_b0, 
    EfficientNet_B0_Weights, 
    resnet18, 
    ResNet18_Weights
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
efficientnet_b0 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
efficientnet_b0.classifier = nn.Identity()
efficientnet_b0.eval()
efficientnet_b0.to(device)

resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18.eval()
resnet18.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def image_to_feature_efficientnet(arr):
    """
    CNN features from numpy array using EfficientNet-B0 (1280 dims)
    """
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert('RGB')

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = efficientnet_b0(img_t).cpu().numpy().flatten()

    return features

def image_to_feature_resnet(arr):
    """
    CNN features from numpy array using ResNet-18 (512 dims)
    """
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert('RGB')

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet18(img_t).cpu().numpy().flatten()

    return features 

def image_to_feature_lbp(arr, P=8, R=1):
    """
    LBP texture features (10 dims, uniform patterns)
    """
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P, R, method='uniform')

    hist, _ = np.histogram(
        lbp,
        bins=P + 2,
        range=(0, P + 2),
        density=True
    )

    return hist 

def image_to_feature_efficientnet_lbp(arr):
    """
    EfficientNet-B0 (1280) + LBP (10) = 1290 dims
    """
    cnn_feat = image_to_feature_efficientnet(arr)
    lbp_feat = image_to_feature_lbp(arr)

    return np.concatenate([cnn_feat, lbp_feat])

def image_to_feature_resnet_lbp(arr):
    """
    ResNet-18 (512) + LBP (10) = 522 dims
    """
    cnn_feat = image_to_feature_resnet(arr)
    lbp_feat = image_to_feature_lbp(arr)

    return np.concatenate([cnn_feat, lbp_feat])

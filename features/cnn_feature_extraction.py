import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# =============================
# Load EfficientNet-B0
# =============================
efficientnet_b0 = models.efficientnet_b0(pretrained=True)

# Remove classifier → feature extractor
efficientnet_b0.classifier = nn.Identity()

efficientnet_b0.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
efficientnet_b0.to(device)

# =============================
# ImageNet preprocessing (EfficientNet-B0)
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # B0 input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# CNN feature extraction
# =============================
def image_to_feature_cnn(arr):
    """
    CNN features from numpy array using EfficientNet-B0 (1280 dims)
    """
    # arr: numpy [H,W,3] in [0,1]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert('RGB')

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = efficientnet_b0(img_t).cpu().numpy().flatten()

    return features  # (1280,)

# =============================
# LBP texture features
# =============================
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

    return hist  # (10,)

# =============================
# CNN + LBP fusion
# =============================
def image_to_feature_cnn_lbp(arr):
    """
    EfficientNet-B0 (1280) + LBP (10) = 1290 dims
    """
    cnn_feat = image_to_feature_cnn(arr)
    lbp_feat = image_to_feature_lbp(arr)

    return np.concatenate([cnn_feat, lbp_feat])

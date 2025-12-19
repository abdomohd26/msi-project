import numpy as np
from skimage import color, filters
from skimage.feature import local_binary_pattern

def image_to_feature_flat(arr):
    """
    Flatten the image to 1D vector
    """
    return arr.flatten()

def image_to_feature_hist_grad(arr, bins=16):
    """
    features: RGB/HSV histograms + gradient stats
    """

    # RGB 
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    r_hist, _ = np.histogram(r, bins=bins, range=(0.0, 1.0), density=True)
    g_hist, _ = np.histogram(g, bins=bins, range=(0.0, 1.0), density=True)
    b_hist, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=True)

    # HSV
    hsv = color.rgb2hsv(arr)
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    v = hsv[:, :, 2].ravel()

    h_hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), density=True)
    s_hist, _ = np.histogram(s, bins=bins, range=(0.0, 1.0), density=True)
    v_hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)

    # Gradient magnitude
    gray = color.rgb2gray(arr)
    grad = filters.sobel(gray)
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)

    # Concatenate: RGB + HSV + gradient stats
    feature = np.concatenate(
        [r_hist, g_hist, b_hist, h_hist, s_hist, v_hist, [grad_mean, grad_std]]
    )
    return feature  # (98 if bins=16)

def image_to_feature_advanced(arr, bins=16):
    """
    features: RGB/HSV histograms + gradient stats + LBP
    """ 

    # RGB 
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    r_hist, _ = np.histogram(r, bins=bins, range=(0.0, 1.0), density=True)
    g_hist, _ = np.histogram(g, bins=bins, range=(0.0, 1.0), density=True)
    b_hist, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=True)

    # HSV 
    hsv = color.rgb2hsv(arr)
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    v = hsv[:, :, 2].ravel()
    h_hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), density=True)
    s_hist, _ = np.histogram(s, bins=bins, range=(0.0, 1.0), density=True)
    v_hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)

    # Gradient stats 
    gray = color.rgb2gray(arr)
    grad = filters.sobel(gray)
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)

    # LBP texture only 
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), density=True)

    # Concat: 48(RGB)+48(HSV)+2(grad)+59(LBP) = 157 dims ✓
    feature = np.concatenate([
        r_hist, g_hist, b_hist, 
        h_hist, s_hist, v_hist, 
        [grad_mean, grad_std],
        lbp_hist
    ])
    
    return feature

import numpy as np

def patchify(img, patch_size):
    h, w = img.shape
    return np.array([
        img[i:i+patch_size, j:j+patch_size]
        for i in range(0, h, patch_size)
        for j in range(0, w, patch_size)
    ])

def unpatchify(patches, img_shape):
    patch_size = patches.shape[1]
    img = np.zeros(img_shape, dtype=patches.dtype)
    patch_idx = 0
    for i in range(0, img_shape[0], patch_size):
        for j in range(0, img_shape[1], patch_size):
            img[i:i+patch_size, j:j+patch_size] = patches[patch_idx]
            patch_idx += 1
    return img


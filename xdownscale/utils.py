import numpy as np

def patchify(imgs, patch_size):
    """
    Supports 2D (y, x) or 3D (n, y, x) inputs.
    Returns all non-overlapping patches of shape (n_patches, patch_size, patch_size)
    """
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, ...]  # convert to (1, y, x)

    n, h, w = imgs.shape
    patches = []

    for img in imgs:
        patches.append(np.array([
            img[i:i+patch_size, j:j+patch_size]
            for i in range(0, h, patch_size)
            for j in range(0, w, patch_size)
        ]))

    return np.concatenate(patches, axis=0)


def unpatchify(patches, img_shape, patch_size):
    """
    Reconstructs image from patches (n_patches, patch_size, patch_size)
    where img_shape is (y, x)
    """
    h, w = img_shape
    img = np.zeros((h, w), dtype=patches.dtype)
    patch_idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            img[i:i+patch_size, j:j+patch_size] = patches[patch_idx]
            patch_idx += 1
    return img

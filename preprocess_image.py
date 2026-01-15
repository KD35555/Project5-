import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)

# ************* ToDo, resize short side *************
def resize_short_side(img_path, target_size=224):
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        # 如果遇到坏图，返回一个全黑的图防止程序崩坏
        return np.zeros((1, 3, target_size, target_size), dtype=np.float32)

    w, h = image.size
    scale = target_size / min(w, h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)

    patch_size = 14
    new_w = int(round(new_w / patch_size) * patch_size)
    new_h = int(round(new_h / patch_size) * patch_size)

    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    image = np.array(image).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = image.transpose(2, 0, 1)
    return image[None]
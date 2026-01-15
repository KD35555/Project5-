import numpy as np

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

cat_pixel_values = center_crop("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values)

dog_pixel_values = center_crop("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values)


try:
    ref_feat = np.load("./demo_data/cat_dog_feature.npy")
    my_feat = np.concatenate([cat_feat, dog_feat], axis=0)
    diff = np.abs(my_feat - ref_feat).mean()
    print(f"Difference: {diff}")
except:
    print("找不到参考文件")

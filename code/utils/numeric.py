import numpy as np
from osgeo import gdal

def calculate_mean_std(image_paths):
    sum_means, sum_squares, count = np.zeros(26), np.zeros(26), 0
    for image_path in image_paths:
        image_ds = gdal.Open(image_path)
        if image_ds is None:
            raise FileNotFoundError(f"Unable to open file: {image_path}")
        image = np.stack([image_ds.GetRasterBand(b).ReadAsArray() for b in range(1, image_ds.RasterCount + 1)], axis=-1)
        sum_means += np.mean(image, axis=(0, 1))
        sum_squares += np.mean(image**2, axis=(0, 1))
        count += 1
    means = sum_means / count
    stds = np.sqrt(sum_squares / count - means**2)
    return means, stds

# 定义反归一化函数
def inverse_normalize(data, mean, std):
    return data * std + mean
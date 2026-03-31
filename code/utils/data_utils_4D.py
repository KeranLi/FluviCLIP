import os
import numpy as np
from osgeo import gdal
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_excel_data(file_path, image_dir, sheet_name):
    """
    从 Excel 文件加载数据并生成文件路径，使用映射表查找第三部分。

    Args:
        file_path (str): Excel 文件路径。
        sheet_name (str): Excel 的工作表名称。
        image_dir (str): 文件存储的根目录。
        mapping_dict (dict): 映射表，key 为样本序号，value 为文件名的第三部分。

    Returns:
        list: 影像路径列表。
        list: 标签列表。
    """
    excel_data = pd.read_excel(file_path, sheet_name=sheet_name)

    # 初始化结果
    shuiwenzhan, shuiwenzhan1, yangpinhao, chulihao, shijian, xuanyizhi, image_paths = [], [], [], [], [], [], []

    k = 0
    for i in range(0, len(excel_data["水文站编号"])):
        if excel_data.at[i, "可行"] == 1:
            shuiwenzhan.append(int(excel_data.at[i, "水文站编号"]))
            shuiwenzhan1.append(str(excel_data.at[i, "水文站名称"]))
            yangpinhao.append(str(excel_data.at[i, "样本序号"]))
            chulihao.append(f"{shuiwenzhan[k]}_{yangpinhao[k]}_{k+1}.tif")
            shijian.append(str(excel_data.at[i, "时间"]))
            xuanyizhi.append((float(excel_data.at[i, "含沙量（g/m3）"])))
            image_paths.append(image_dir + f"{shuiwenzhan[k]}_{yangpinhao[k]}_{k+1}.tif")
            k += 1

    return image_paths, xuanyizhi

class Sentinel2Dataset(Dataset):
    def __init__(self, image_paths, labels, means=None, stds=None):
        self.image_paths = image_paths
        self.labels = labels  # 添加这一行来初始化标签
        #self.transform = transform
        self.means = means
        self.stds = stds
        self.label_mean = np.mean(labels)
        self.label_std = np.std(labels)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取影像数据
        image_path = self.image_paths[idx]
        image_ds = gdal.Open(image_path)
        if image_ds is None:
            raise FileNotFoundError(f"Unable to open file: {image_path}")

        # 读取所有波段
        image = []
        for b in range(1, image_ds.RasterCount + 1):
            band = image_ds.GetRasterBand(b)
            image.append(band.ReadAsArray())
        image = np.stack(image, axis=-1)  # shape: (H, W, C)

        # 转换为 Tensor 并调整为 (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # 转换为 Tensor
        
        # 调整大小到 (224, 224) 使用 torch.nn.functional.interpolate
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # 应用归一化
        # 是要先归一化再插值呢？还是先插值再归一化？
        if self.means is not None and self.stds is not None:
            for i in range(image.shape[0]):
                if self.stds[i] == 0:
                    self.stds[i] = 1e-6  # 设置为一个很小的正数
                image[i] = (image[i] - self.means[i]) / self.stds[i]

        # 获取标签
        label = self.labels[idx]
        label = (label - self.label_mean) / self.label_std  # 归一化标签
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
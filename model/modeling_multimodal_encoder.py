import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

class DummyMultimodalDataset(Dataset):
    """
    演示用的 Dummy 数据集，每次返回:
      - raw_img: 一张 PIL.Image
      - text: 一个字符串
      - attr_imgs: 若干属性图像(PIL)列表
      - label: 一个分类标签 (int)
    """
    def __init__(self, num_samples=100, num_classes=5):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) 随机生成图像(使用纯色方块来模拟)
        raw_img = Image.new('RGB', (224, 224), color=(random.randint(0,255),
                                                      random.randint(0,255),
                                                      random.randint(0,255)))
        # 2) 随机文字
        text = "This is a dummy text {}".format(idx)
        
        # 3) 属性图像列表 (这里简单用2张)
        attr_img1 = Image.new('RGB', (224, 224), color=(random.randint(0,255),
                                                        random.randint(0,255),
                                                        random.randint(0,255)))
        attr_img2 = Image.new('RGB', (224, 224), color=(random.randint(0,255),
                                                        random.randint(0,255),
                                                        random.randint(0,255)))
        attr_imgs = [attr_img1, attr_img2]
        
        # 4) 随机标签
        label = random.randint(0, self.num_classes - 1)
        
        return raw_img, text, attr_imgs, label
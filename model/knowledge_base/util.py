import cv2, json, re
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET
from openai import OpenAI
from data.build_data.voc import PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_ALL_CATEGORIES

def extract_horses_pixels(image_path, mask_path):
    # 读取原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取分割掩码
    mask = Image.open(mask_path)
    mask = np.array(mask)
    # 统计不同数字的个数
    unique, counts = np.unique(mask, return_counts=True)

    # 打印结果
    for u, c in zip(unique, counts):
        print(f"数字 {u}: {c} 个")
    horses_mask = (mask == 13).astype(np.uint8) * 255
    
    # 生成 RGBA 图像，背景设为透明
    alpha_channel = horses_mask.copy()
    cropped_image = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel])

    return cropped_image, horses_mask


class VOCDataLoader:
    def __init__(self, root_dir, split_id):
        self.root_dir = root_dir
        self.split_id = split_id
        self.class_info = self._init_class_mapping()
        self.mask_palette = self._init_mask_palette()
    
    def _init_class_mapping(self):
        """VOC语义分割颜色映射（RGB格式）"""
        return {
            'aeroplane': (128, 0, 0),
            'bicycle': (0, 128, 0),
            'bird': (128, 128, 0),
            'boat': (0, 0, 128),
            'bottle': (128, 0, 128),
            'bus': (0, 128, 128),
            'car': (128, 128, 128),
            'cat': (64, 0, 0),
            'chair': (192, 0, 0),
            'cow': (64, 128, 0),
            'diningtable': (192, 128, 0),
            'dog': (64, 0, 128),
            'horse': (192, 0, 128),
            'motorbike': (64, 128, 128),
            'person': (192, 128, 128),
            'pottedplant': (0, 64, 0),
            'sheep': (128, 64, 0),
            'sofa': (0, 192, 0),
            'train': (128, 192, 0),
            'tvmonitor': (0, 64, 128)
        }
    
    def _init_mask_palette(self):
        """生成颜色到类别ID的查找表"""
        palette = {}
        for idx, (cls_name, color) in enumerate(self.class_info.items()):
            palette[color] = idx
        return palette
    
    def _load_class_info(self):
        """加载VOC类别与分割ID的映射关系"""
        return {
            # 示例映射，实际需要根据VOC官方定义
            'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
            'bottle':4, 'bus':5, 'car':6, 'cat':7, 'chair':8,
            'cow':9, 'diningtable':10, 'dog':11, 'horse':12,
            'motorbike':13, 'person':14, 'pottedplant':15,
            'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
        }
    
    def get_annotation_path(self, img_id):
        return os.path.join(self.root_dir, 'Annotations', f'{img_id}.xml')
    
    def get_base_classes(self):
        """获取当前split的base classes"""
        return PASCAL_VOC_BASE_CATEGORIES[self.split_id]

    def get_all_classes(self):
        """获取所有的classes"""
        return PASCAL_VOC_ALL_CATEGORIES[self.split_id]

    def get_image_list(self):
        """获取当前split的训练集列表"""
        with open(os.path.join(self.root_dir, 'ImageSets/Main/train.txt')) as f:
            return [x.strip() for x in f.readlines()]

    def load_image(self, img_id):
        """加载图像和对应的分割标注"""
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{img_id}.jpg')
        
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"警告：图像文件 {img_path} 不存在")
            return None, None
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像 {img_id} 失败: {str(e)}")
            return None, None
        
        return image

    def extract_class_region(self, image, mask, class_name):
        """提取指定类别的有效区域(矩形区域)"""
        class_id = self.class_info[class_name]
        class_mask = (mask == class_id).astype(np.uint8)
        
        if np.sum(class_mask) == 0:
            return None, None
            
        # 寻找有效区域的最小外接矩形
        rows, cols = np.where(class_mask)
        min_y, max_y = np.min(rows), np.max(rows)
        min_x, max_x = np.min(cols), np.max(cols)
        
        cropped_image = image[min_y:max_y+1, min_x:max_x+1]
        cropped_mask = class_mask[min_y:max_y+1, min_x:max_x+1]
        
        return cropped_image, cropped_mask
    
    def extract_class_mask(self, image, semantic_mask, class_name):
        """
        提取像素级掩码并生成RGBA图像
        :param image: PIL.Image格式的原始图像
        :param semantic_mask: 语义分割掩码图像（P模式）
        :param class_name: 目标类别名称
        :return: (RGBA图像, 二值掩码) 或 (None, None)
        """
        # 转换掩码为numpy数组
        mask_array = np.array(semantic_mask)
        
        # 获取目标类别颜色
        target_color = self.class_info.get(class_name)
        if target_color is None:
            return None, None

        # 生成二值掩码
        binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
        target_pixels = (mask_array == target_color)
        if np.sum(target_pixels) == 0:
            return None, None
        
        binary_mask[target_pixels] = 255

        # 生成RGBA图像
        np_image = np.array(image.convert("RGB"))
        rgba_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2RGBA)
        rgba_image[:, :, 3] = binary_mask  # Alpha通道设为掩码

        return Image.fromarray(rgba_image), binary_mask

class AttributeManager:
    def __init__(self, split_id, cache_dir="/root/fsod/model/knowledge_base/attribute_cache"):
        self.split_id = split_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / f"voc_attributes_split{split_id}.json"
        
        # 加载已有缓存
        self.attributes = self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def generate_all(self, class_list, generator):
        """批量生成所有缺失的属性"""
        missing = [c for c in class_list if c not in self.attributes]
        if not missing:
            return

        print(f"Generating attributes for {len(missing)} classes...")
        for cls in tqdm(missing):
            try:
                self.attributes[cls] = generator.generate_attributes(cls)
            except Exception as e:
                print(f"Failed to generate {cls}: {str(e)}")
        
        self._save_cache()

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.attributes, f, indent=2)

    def get_attribute(self, class_name):
        """获取指定类别的属性描述"""
        return self.attributes.get(class_name, None)

    @classmethod
    def validate_attributes(cls, attributes):
        """验证属性格式有效性"""
        valid = {}
        for k, v in attributes.items():
            if ':' in v and len(v.split(':')) > 1:
                valid[k] = v
            else:
                print(f"Invalid format for {k}: {v}")
        return valid
    
class AttributeGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.format_checker = re.compile(r"^[a-zA-Z]+: [a-zA-Z, ]+$")

    def generate_attributes(self, class_name):
        """生成属性并验证格式"""
        response = self._safe_api_call(class_name)
        return self._format_response(class_name, response)

    def _safe_api_call(self, class_name, max_retries=3):
        for _ in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{
                        "role": "user",
                        "content": self._build_prompt(class_name)
                    }]
                )
            except Exception as e:
                print(f"Retrying {class_name} due to error: {str(e)}")
        raise Exception(f"Failed to generate attributes for {class_name}")

    def _build_prompt(self, class_name):
        return f"""Describe the visual appearance of a {class_name} focusing on distinctive parts and textures. 
        Use exactly this format: {class_name}: [noun phrase1], [noun phrase2], ... 
        Example: horse: muscular torso, long mane, flowing tail, slender legs, hard hooves"""

    def _format_response(self, class_name, response):
        raw_text = response.choices[0].message.content.strip()
        
        # 格式标准化处理
        if raw_text.startswith(class_name):
            return raw_text
        
        # 处理可能的多余前缀
        clean_text = raw_text.replace(f"The {class_name} ", "")\
                             .replace(f"A {class_name} ", "")\
                             .replace("Answer: ", "")
        return f"{class_name}: {clean_text.split(':')[-1].strip()}"
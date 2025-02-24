import os
import cv2
import numpy as np
import torch
import clip
import joblib

from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from urllib.request import urlretrieve
import hashlib

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.segmentation import slic, felzenszwalb, mark_boundaries
from skimage.util import img_as_float

class Knowledge_Base:
    """
    知识库主类，同时支持三种过分割算法：
      - SAM (Segment Anything Model)
      - SLIC (基于 skimage.segmentation.slic)
      - FELZENSZWALB (基于 skimage.segmentation.felzenszwalb)
    """

    def __init__(self, 
                 device: str = None, 
                 clip_model: str = 'ViT-B/32', 
                 sam_model_type: str = "vit_h", 
                 segmentation_method: str = "SLIC",
                 similarity_threshold: float = 0.22,
                 segmentation_params: dict = None):
        """
        :param device: 设备，如 "cuda" 或 "cpu"
        :param clip_model: CLIP 模型类型（如 'ViT-B/32'）
        :param sam_model_type: SAM 模型类型（'vit_h'、'vit_l'、'vit_b'）
        :param segmentation_method: 选择使用的过分割算法 { "SAM" | "SLIC" | "FELZENSZWALB" }
        :param similarity_threshold: CLIP 匹配时的相似度阈值
        :param segmentation_params: 分割参数 (字典形式)，若为空则使用默认值
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(clip_model, self.device)
        print(f'Using {segmentation_method} algorithm to segmentate')

        # knowledge 结构：{class_name: {"text_embeds": None, "image_embeds": {attr_name: [ {embedding, patch, similarity}, ... ]}}}
        self.knowledge = defaultdict(lambda: {
            "text_embeds": None,
            "image_embeds": defaultdict(list)
        })

        # 用于缓存文本：{ class_name: { overall: Tensor, class: Tensor, attributes: [Tensor], attribute_names: [str] } }
        self.text_cache = {}

        # 查询用的索引缓存：{ class_name: { "embeddings": np.ndarray, "mapping": [(attr, idx), ...] } }
        self.index_cache = {}

        # 选择分割算法
        assert segmentation_method in ["SAM", "SLIC", "FELZENSZWALB"], \
            f"segmentation_method must be one of ['SAM','SLIC','FELZENSZWALB'], but got {segmentation_method}"
        self.segmentation_method = segmentation_method

        # 相似度阈值
        self.similarity_threshold = similarity_threshold

        # 初始化 SAM 相关（若选择 SAM）
        self.sam_model = None
        self.mask_generator = None
        self.default_sam_params = {
            "points_per_side": 64,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.88,
            "crop_n_layers": 2,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 450,
            "box_nms_thresh": 0.6
        }

        # 初始化 SLIC / FELZENSZWALB 默认参数
        self.default_slic_params = {
            "n_segments": 10,
            "compactness": 5.0,
            "sigma": 1.0,
            "start_label": 1
        }
        self.default_felzen_params = {
            "scale": 500,
            "sigma": 0.6,
            "min_size": 100
        }

        # 合并并存储分割参数
        self.seg_params = segmentation_params or {}
        if self.segmentation_method == "SAM":
            # SAM 的默认参数合并
            self.seg_params = {**self.default_sam_params, **self.seg_params}
            # 初始化 SAM 模型
            self.sam_model = self._init_sam(sam_model_type)
            self._update_mask_generator()
        elif self.segmentation_method == "SLIC":
            # SLIC 的默认参数合并
            self.seg_params = {**self.default_slic_params, **self.seg_params}
        elif self.segmentation_method == "FELZENSZWALB":
            # FELZENSZWALB 的默认参数合并
            self.seg_params = {**self.default_felzen_params, **self.seg_params}

    def _parse_bbox_annotations(self, xml_file: str, class_name: str) -> List[Tuple[int, int, int, int]]:
        """
        从VOC的XML文件中解析出指定类别的目标bbox。
        :param xml_file: VOC的xml文件路径
        :param class_name: 类别名称（例如 "horse"）
        :return: 指定类的bbox列表（每个bbox为(xmin, ymin, xmax, ymax)）
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        bboxes = []
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name == class_name:  # 如果类别匹配
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                bboxes.append((xmin, ymin, xmax, ymax))
        
        return bboxes

    def _crop_bboxes(self, image: Image.Image, bboxes: List[Tuple[int, int, int, int]]) -> List[Image.Image]:
        """
        根据给定的bbox坐标裁剪图像，返回裁剪出的patches。
        :param image: 原始图像
        :param bboxes: 每个目标的bbox (xmin, ymin, xmax, ymax)
        :return: 裁剪出的多个patch
        """
        patches = []
        for (xmin, ymin, xmax, ymax) in bboxes:
            patch = image.crop((xmin, ymin, xmax, ymax))
            patches.append(patch)
        return patches

    def _init_sam(self, model_type: str):
        """
        自动检测并下载指定类型的 SAM 权重，并加载到内存中。
        """
        model_config = {
            "vit_h": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "md5": "4b8939a88964f0f4ff5f5b2642c598a6",
                "path": "./weights/sam_vit_h_4b8939.pth"
            },
            "vit_l": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "md5": "0b3195f0f5eb5f2a102916e6185a4e3a",
                "path": "./weights/sam_vit_l_0b3195.pth"
            },
            "vit_b": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "md5": "01ec64d29a2fca3f334193cf5b535a03",
                "path": "./weights/sam_vit_b_01ec64.pth"
            }
        }

        if model_type not in model_config:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from {list(model_config.keys())}")

        cfg = model_config[model_type]
        os.makedirs(os.path.dirname(cfg["path"]), exist_ok=True)

        if not self._check_file(cfg["path"], cfg["md5"]):
            print(f"Downloading SAM weights ({model_type}) ...")
            self._download_with_progress(cfg["url"], cfg["path"])
            if not self._check_file(cfg["path"], cfg["md5"]):
                raise RuntimeError(f"Failed to download valid {model_type} model. "
                                   f"Please check or download manually from {cfg['url']}")

        sam = sam_model_registry[model_type](checkpoint=cfg["path"])
        sam.to(device=self.device)
        return sam

    def _check_file(self, path: str, expected_md5: str) -> bool:
        """检查文件完整性和MD5"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                md5val = hashlib.md5(f.read()).hexdigest()
            return md5val == expected_md5
        except Exception:
            return False

    def _download_with_progress(self, url: str, filename: str):
        """带进度条的下载方法"""
        try:
            from tqdm import tqdm
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(filename)) as t:
                urlretrieve(url, filename=filename, reporthook=t.update_to)
        except ImportError:
            print("tqdm not installed, downloading without progress bar...")
            urlretrieve(url, filename=filename)

    def _update_mask_generator(self):
        """基于当前 SAM 参数重新创建分割器"""
        if self.sam_model is None:
            return
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model, **self.seg_params)

    def update_segmentation_params(self, new_params: dict):
        """
        动态更新分割参数并重新初始化分割器
        :param new_params: 需要更新的参数字典
        """
        if self.segmentation_method == "SAM":
            self.seg_params = {**self.seg_params, **new_params}
            self._update_mask_generator()
            print(f"[SAM] segmentation params updated: {self.seg_params}")
        elif self.segmentation_method == "SLIC":
            self.seg_params = {**self.seg_params, **new_params}
            print(f"[SLIC] segmentation params updated: {self.seg_params}")
        elif self.segmentation_method == "FELZENSZWALB":
            self.seg_params = {**self.seg_params, **new_params}
            print(f"[FELZENSZWALB] segmentation params updated: {self.seg_params}")

    def _parse_attribute_text(self, text: str) -> Tuple[str, List[str]]:
        """
        将类似于 "Horses: a muscular build, a long mane, ..." 的文本解析为:
        - class_name: "Horses"
        - [class_name, 'a muscular build', 'a long mane', ...]
        """
        class_name, attributes = text.split(":", 1)
        class_name = class_name.strip()
        # 原文为第0个embedding，将类名本身放在第1个embedding，后续全部是属性embedding
        components = [text] + [class_name] + [attr.strip()+f' of a {class_name}' for attr in attributes.strip(" .").split(", ")]
        return class_name, components

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """批量文本编码"""
        with torch.no_grad():
            text_inputs = clip.tokenize(texts).to(self.device)
            return self.model.encode_text(text_inputs)

    def process_attributes(self, attribute_text: str) -> Dict[str, torch.Tensor]:
        """
        处理属性文本，得到:
        {
            "overall": embeddings[0],  # 整句 Embedding
            "class": embeddings[1],    # 类名 Embedding
            "attributes": [后续属性],
            "attribute_names": [属性文本列表]
        }
        """
        class_name, components = self._parse_attribute_text(attribute_text)
        embeddings = self._encode_texts(components)

        self.text_cache[class_name] = {
            "overall": embeddings[0],
            "class": embeddings[1],
            "attributes": [embeddings[i] for i in range(2, len(components))],
            "attribute_names": components[2:]
        }
        return self.text_cache[class_name]

    def _segment_image_sam(self, image: Image.Image) -> List[Image.Image]:
        """
        使用 SAM 进行分割，返回若干 patches (PIL.Image)。
        """
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_size = cv_image.shape[:2]

        # 长边不超过1024
        scale_factor = max(1024 / max(original_size), 1.0)
        new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
        resized_image = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_LINEAR)

        with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu"):
            masks = self.mask_generator.generate(resized_image)

        # 过滤并根据面积降序排序
        valid_masks = [
            m for m in sorted(masks, key=lambda x: x["area"], reverse=True)
            if m['area'] > 500 and m['predicted_iou'] > 0.9
        ]

        patches = []
        for mask in valid_masks:
            # 将掩码缩放回原图大小
            mask_array = cv2.resize(
                mask["segmentation"].astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            # 提取有效区域
            masked = cv2.bitwise_and(
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                mask=mask_array
            )
            y_coords, x_coords = np.where(mask_array)
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            patch = masked[y_min:y_max+1, x_min:x_max+1]
            patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            patches.append(patch_pil)

        return patches

    def _segment_image_slic(self, image: Image.Image) -> List[Image.Image]:
        """
        使用 SLIC 进行过分割，并将每个超像素区域裁剪成 patch (PIL.Image) 返回。
        """
        # 转 numpy
        np_image = np.array(image.convert("RGB"))
        float_image = img_as_float(np_image)
        
        # 进行 SLIC 分割
        segments = slic(
            float_image,
            n_segments=self.seg_params.get("n_segments", 10),
            compactness=self.seg_params.get("compactness", 5.0),
            sigma=self.seg_params.get("sigma", 1.0),
            start_label=self.seg_params.get("start_label", 1)
        )

        # 根据 segments 标签裁剪每个区域
        patches = []
        for seg_val in np.unique(segments):
            if seg_val == 0:  # 忽略背景
                continue
                
            # 生成当前区域的二值掩码
            segment_mask = (segments == seg_val)
            ys, xs = np.where(segment_mask)
            
            # 计算最小外接矩形
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            
            # 裁剪区域
            patch = np_image[y_min:y_max+1, x_min:x_max+1]
            patches.append(Image.fromarray(patch))
            
        return patches
    
    def _segment_image_slic_multi_scale(self, image: Image.Image) -> List[Image.Image]:
        """
        使用不同 SLIC 参数进行多尺度分割，返回不同尺度的图像块列表。
        :param image: 输入图像
        :return: 不同尺度的图像块列表
        """
        np_image = np.array(image.convert("RGB"))
        float_image = img_as_float(np_image)

        # 设置多尺度参数
        scales = {
            "large": {"n_segments": 13, "compactness": 10, "sigma": 0.2, "start_label": 1},
            "medium": {"n_segments": 20, "compactness": 5, "sigma": 0.2, "start_label": 1},
            "small": {"n_segments": 30, "compactness": 5, "sigma": 0.2, "start_label": 1},
        }

        segments = {}
        patches = []
        
        # 分别使用大、中、小尺度参数进行分割
        for scale, params in scales.items():
            segments[scale] = slic(
                float_image,
                n_segments=params["n_segments"],
                compactness=params["compactness"],
                sigma=params["sigma"]
            )
        
        '''
        save_dir = '/data/lihl/fsod/model/knowledge_base/debug_sam/slic_result'
        # 保存每个尺度的分割结果
        for scale, segment in segments.items():
            # 使用mark_boundaries绘制分割边界
            boundary_image = mark_boundaries(
                np.array(image), segment, color=(1, 0, 0), mode='thick'
            )

            # 创建图像文件名
            save_path = os.path.join(save_dir, f"slic_{scale}_.png")
            # 保存带边界线的图像
            plt.imsave(save_path, boundary_image)

            #print(f"Saved SLIC results in {save_path}")
        '''
        # 生成每个区域的图像块
        for scale in segments:
            unique_segments = np.unique(segments[scale])
            for seg_val in unique_segments:
                if seg_val == 0:  # 忽略背景
                    continue

                # 生成当前区域的二值掩码
                segment_mask = (segments[scale] == seg_val)
                ys, xs = np.where(segment_mask)

                # 计算最小外接矩形
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)

                # 裁剪区域并存入结果列表
                patch = np_image[y_min:y_max + 1, x_min:x_max + 1]
                patches.append(Image.fromarray(patch))

        return patches
    

    def visualize_and_save_slic_multiscale(self, image: Image.Image, class_name: str, save_dir: str, xml_file: str, top_k: int = 5):
        """
        可视化并保存不同尺度的 SLIC 分割结果，同时保存每个图像块。
        :param image: 输入图像
        :param class_name: 类别名
        :param save_dir: 保存图像的目录
        :param top_k: 保存的前K个结果
        """
        # 从XML文件中获取目标的bbox标注
        bboxes = self._parse_bbox_annotations(xml_file, class_name)
        # 根据bbox裁剪图像
        patches = self._crop_bboxes(image, bboxes)

        # 步骤1: 根据 bbox 提取物体区域（patches）
        patches = self._crop_bboxes(image, bboxes)  # 根据bbox裁剪出多个物体区域
        if not patches:
            return {}

        # 步骤2: 对每个裁剪出的物体图像（patch）进行分割
        all_fine_grained_patches = []
        for patch in patches:
            fine_grained_patches = self._segment_image_slic_multi_scale(patch)  # 对每个 patch 进行细粒度分割
            all_fine_grained_patches.extend(fine_grained_patches)

        # 确保保存路径存在
        os.makedirs(save_dir, exist_ok=True)

        # 文本属性数据
        text_data = self.text_cache.get(class_name)
        if not text_data or not text_data["attributes"]:
            print("No attributes found for class", class_name)
            return

        # 图像编码
        patches_saved = []
        for i, patch in enumerate(all_fine_grained_patches):
            # 进行图像的 CLIP 编码
            image_embed = self._encode_images([patch])
            
            # 获取最相似的属性
            attribute_tensor = torch.stack(text_data["attributes"]).to(self.device)
            matches, similarity_matrix = self._find_best_matches(image_embed, attribute_tensor)

            # 构造属性文本和相似度
            if matches[0] != -1:  # 如果有匹配
                attr_name = text_data["attribute_names"][matches[0]]
                sim_score = similarity_matrix[0, matches[0]].item()

                # 将图像块保存为文件
                file_name = f"{class_name}_{attr_name}_patch_{i}_sim_{sim_score:.2f}.png"
                save_path = os.path.join(save_dir, file_name)
                patch.save(save_path)

                # 保存时带上标题，方便查看
                plt.imshow(patch)
                plt.title(f"{attr_name}: {sim_score:.2f}")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()

                patches_saved.append(save_path)

        print(f"Saved {len(patches_saved)} patches to {save_dir}")

    def _segment_image_felzenszwalb(self, image: Image.Image) -> List[Image.Image]:
        """
        使用 FELZENSZWALB 算法进行过分割，并将每个区域裁剪成 patch (PIL.Image)。
        """
        np_image = np.array(image.convert("RGB"))
        float_image = img_as_float(np_image)
        segments = felzenszwalb(
            float_image,
            scale=self.seg_params.get("scale", 100),
            sigma=self.seg_params.get("sigma", 0.5),
            min_size=self.seg_params.get("min_size", 50)
        )

        # 构造每个区域的 patches
        patches = []
        unique_segments = np.unique(segments)
        for seg_val in unique_segments:
            mask = (segments == seg_val).astype(np.uint8)
            ys, xs = np.where(mask == 1)
            if len(xs) < 1 or len(ys) < 1:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            region = np_image[y_min:y_max+1, x_min:x_max+1]
            patches.append(Image.fromarray(region, mode="RGB"))

        return patches

    def _segment_image(self, image: Image.Image, **kwargs) -> List[Image.Image]:
        """
        根据 self.segmentation_method 选择合适的过分割算法，返回分割后的小图列表。
        """
        if self.segmentation_method == "SAM":
            return self._segment_image_sam(image, **kwargs)
        elif self.segmentation_method == "SLIC":
            return self._segment_image_slic_multi_scale(image)
            return self._segment_image_slic(image, **kwargs)
        elif self.segmentation_method == "FELZENSZWALB":
            return self._segment_image_felzenszwalb(image, **kwargs)
        else:
            raise ValueError("Unknown segmentation method.")

    def _encode_images(self, patches: List[Image.Image]) -> torch.Tensor:
        """批量图像编码，返回图像特征"""
        if not all(isinstance(p, Image.Image) for p in patches):
            raise ValueError("All patches must be PIL.Image instances")
        with torch.no_grad():
            image_inputs = torch.stack([self.preprocess(p) for p in patches]).to(self.device)
            return self.model.encode_image(image_inputs)

    def _find_best_matches(self, 
                           image_embeds: torch.Tensor, 
                           text_embeds: torch.Tensor
                           ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        根据相似度矩阵寻找最佳匹配：
          - matches: (N,) 长度，其中每个元素是匹配的 text 索引，若无效匹配则为 -1
          - similarity_matrix: 原始的（未缩放）相似度矩阵
        """
        # 归一化
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # 计算相似度矩阵
        similarity_matrix = image_embeds @ text_embeds.T

        # CLIP 的温度参数
        logit_scale = self.model.logit_scale.exp()
        scaled_similarity = similarity_matrix * logit_scale

        # 在 scaled_similarity 上做最大值匹配
        max_sim = scaled_similarity.max(dim=1).values
        valid_mask = max_sim > self.similarity_threshold

        # 构造匹配索引
        matches = torch.full(
            (scaled_similarity.size(0),), 
            -1, 
            dtype=torch.long, 
            device=image_embeds.device
        )
        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() > 0:
            matches[valid_indices] = torch.argmax(scaled_similarity[valid_indices], dim=1)

        return matches.cpu().numpy(), similarity_matrix.cpu()

    def process_image(self, 
                    image: Image.Image, 
                    xml_file: str, 
                    class_name: str, 
                    save_patches: bool = False,
                    **kwargs) -> Dict:
        """
        处理图像，使用bbox标注进行裁剪，之后进行CLIP匹配。
        :param image: 需要处理的图像
        :param xml_file: VOC格式的xml标注文件
        :param class_name: 类别名
        :param save_patches: 是否保存分割结果
        :return: 属性文本和图像匹配结果
        """
        if class_name not in self.text_cache:
            raise ValueError(f"Class {class_name} not found in text_cache. Please process_attributes first.")
        
        # 从XML文件中获取目标的bbox标注
        bboxes = self._parse_bbox_annotations(xml_file, class_name)
        # 根据bbox裁剪图像
        patches = self._crop_bboxes(image, bboxes)
        if not patches:
            return {}
        # 步骤2: 对每个裁剪出的物体图像（patch）进行分割
        all_fine_grained_patches = []
        for patch in patches:
            # 这里的fine_grained_patches是 矩形框 的形式
            fine_grained_patches = self._segment_image(patch)  # 对每个 patch 进行细粒度分割
            all_fine_grained_patches.extend(fine_grained_patches)
        # 步骤3: 图像编码
        image_embeds = self._encode_images(all_fine_grained_patches)

        # 文本属性
        text_data = self.text_cache[class_name]
        if not text_data["attributes"]:
            return {}

        attribute_tensor = torch.stack(text_data["attributes"]).to(self.device)
        matches, similarity_matrix = self._find_best_matches(image_embeds, attribute_tensor)

        # 构造结果
        results = defaultdict(list)
        
        # 逐个处理每个匹配结果
        for idx, match_idx in enumerate(matches):
            if match_idx == -1:
                continue
            attr_name = text_data["attribute_names"][match_idx]
            sim_score = similarity_matrix[idx, match_idx].item()
            
            # 如果相似度大于阈值，将其加入结果
            if sim_score >= self.similarity_threshold:
                results[attr_name].append({
                    "embedding": image_embeds[idx].cpu().clone(),
                    "patch": all_fine_grained_patches[idx],
                    "similarity": sim_score
                })

        # 按相似度降序排列每个属性的匹配结果
        for attr in results:
            results[attr].sort(key=lambda x: x["similarity"], reverse=True)

        return results

    def add_to_knowledge(self, class_name: str, image_results: Dict):
        """
        将 process_image 的结果加入到知识库中。
        """
        if class_name not in self.text_cache:
            raise ValueError(f"Class {class_name} not in text_cache.")

        for attr_name, items in image_results.items():
            self.knowledge[class_name]["image_embeds"][attr_name].extend(items)

        # 索引缓存失效
        if class_name in self.index_cache:
            del self.index_cache[class_name]

    def _build_index(self, class_name: str):
        """
        为指定类构建或更新索引，以加速 query。
        """
        all_embeds = []
        mapping = []
        for attr, items in self.knowledge[class_name]["image_embeds"].items():
            for idx, item in enumerate(items):
                all_embeds.append(item["embedding"].numpy())
                mapping.append((attr, idx))

        if len(all_embeds) == 0:
            return

        self.index_cache[class_name] = {
            "embeddings": np.stack(all_embeds),
            "mapping": mapping
        }

    def query(self, 
              query_embed: torch.Tensor, 
              class_name: str, 
              top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        在知识库中检索最相似的属性：
          返回: [ (attr_name, similarity, index_in_attr_list), ... ]
        """
        if class_name not in self.index_cache:
            self._build_index(class_name)
        if class_name not in self.index_cache or "embeddings" not in self.index_cache[class_name]:
            return []

        index_data = self.index_cache[class_name]
        query_np = query_embed.cpu().numpy()

        # 余弦相似度
        norms = np.linalg.norm(index_data["embeddings"], axis=1)
        query_norm = np.linalg.norm(query_np)
        similarities = (index_data["embeddings"] @ query_np) / (norms * query_norm)

        top_indices = np.argsort(-similarities)[:top_k]
        results = []
        for i in top_indices:
            attr_name, idx_in_attr = index_data["mapping"][i]
            sim = float(similarities[i])
            results.append((attr_name, sim, idx_in_attr))
        return results
    
    def generate_refer_feature_map(self, 
                                query_image: Image.Image, 
                                top_k: int = 5,
                                grid_sizes=[2,4,6]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        生成参考特征图的完整方法
        :param query_image: 查询图像 (PIL.Image)
        :param top_k: 每个patch查询的top_k相似度结果
        :param grid_sizes: 网格划分策略，默认[2,4,6]对应2x2,4x4,6x6
        :return: (属性文本embedding列表, 图像特征矩阵 [num_patches, 512])
        """
        if not isinstance(query_image, Image.Image):
            raise TypeError("query_image 必须是 PIL.Image 类型")
        if query_image.mode != "RGB":
            query_image = query_image.convert("RGB")

        # 用于保存选定属性对应的文本 Embedding
        all_text_embeds = []
        # 用于保存图像 patch 的图像 Embedding
        all_image_embeds = []
        count = 0

        # 先对整张图做一次细粒度分割
        patches = self._segment_image(query_image)

        # 处理每个分割得到的 patch
        for patch in patches:
            # 编码该 patch（图像特征）
            with torch.no_grad():
                patch_tensor = self.preprocess(patch).unsqueeze(0).to(self.device)
                patch_embed = self.model.encode_image(patch_tensor).squeeze(0)

            # 在知识库里对所有已知类别做查询
            candidates = []
            for cls_name in self.knowledge.keys():
                # self.query 返回: [(attr_name, similarity, index_in_attr_list), ...]
                results = self.query(patch_embed, cls_name, top_k=top_k)
                if results:
                    # 把 (cls_name, (attr_name, sim, idx)) 放到 candidates
                    candidates.extend([(cls_name, res) for res in results])

            if candidates:
                # 根据 similarity 做一个随机加权采样(也可以直接取相似度最高的)
                similarities = [r[1] for _, r in candidates]  # r[1] 是 sim
                probs = torch.softmax(torch.tensor(similarities), dim=0).numpy()
                selected_idx = np.random.choice(len(candidates), p=probs)
                selected_cls, (attr_name, sim, _) = candidates[selected_idx]

                # === 这里改为取文本 embedding ===
                # 从 text_cache 中找出 attr_name 对应的文本向量
                text_idx = self.text_cache[selected_cls]["attribute_names"].index(attr_name)
                text_emb = self.text_cache[selected_cls]["attributes"][text_idx].cpu()

                # 分别存入列表
                all_text_embeds.append(text_emb)
                all_image_embeds.append(patch_embed.cpu())

                # 若需要可视化
                '''                
                self.visualize_query(
                    query_patch=patch,
                    top_k=5,
                    results=results,
                    class_name=selected_cls,
                    save_path=f'/data/lihl/fsod/debug/query_result_patch_{count}.png'
                )
                count += 1
                '''
            else:
                print(f"Warning: No suitable match found for this patch. Use zero vectors instead.")
                # 给文本向量和图像向量都放一个全零
                all_text_embeds.append(torch.zeros(512))
                all_image_embeds.append(torch.zeros(512))

        # 将图像 embedding 列表拼成一个 (N, 512) 的张量
        refer_feature_map = torch.stack(all_image_embeds, dim=0)

        # 返回： (属性文本的embedding列表, 图像特征矩阵)
        return all_text_embeds, refer_feature_map

    def _grid_split_image(self, image: Image.Image, grid_size: int) -> List[Image.Image]:
        """
        将图像按grid_size x grid_size均匀分割
        :return: 按行优先顺序排列的patch列表
        """
        img_w, img_h = image.size
        
        # 计算每个块的尺寸（允许最后一块尺寸不同）
        patch_w = img_w // grid_size
        patch_h = img_h // grid_size

        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算裁剪区域
                left = j * patch_w
                upper = i * patch_h
                right = (j+1)*patch_w if j != grid_size-1 else img_w
                lower = (i+1)*patch_h if i != grid_size-1 else img_h
                
                # 裁剪并保存
                patch = image.crop((left, upper, right, lower))
                patches.append(patch)

        print(f'len(patches): {len(patches)}')
        return patches

    def visualize_patches(self, image: Image.Image, spacing=10, background=(255,255,255), save_path: str = None):
        """
        改进的可视化方法，带间隔
        :param spacing: 间隔像素数
        :param background: 背景颜色 (RGB)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.05)  # 减少子图间距

        for ax_idx, grid_size in enumerate([2,4,6]):
            # 生成带间隔的网格图像
            grid_img = self._arrange_grid_with_spacing(
                image=image,
                grid_size=grid_size,
                spacing=spacing,
                background=background
            )
            axes[ax_idx].imshow(grid_img)
            axes[ax_idx].set_title(f"{grid_size}x{grid_size}", fontsize=10, pad=8)
            axes[ax_idx].axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

    def _arrange_grid_with_spacing(self, image: Image.Image, grid_size: int, spacing: int, background: tuple) -> Image.Image:
        """带间隔的网格排列方法"""
        patches = self._grid_split_image(image, grid_size)
        sample_patch = patches[0]
        pw, ph = sample_patch.size
        
        canvas_w = grid_size * pw + (grid_size - 1) * spacing
        canvas_h = grid_size * ph + (grid_size - 1) * spacing
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=background)
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * (pw + spacing)
                y = i * (ph + spacing)
                canvas.paste(patches[i*grid_size + j], (x, y))
        
        return canvas

    def save(self, filename: str):
        """
        保存知识库到文件
        """
        knowledge_data = {
            "knowledge": dict(self.knowledge),
            "text_cache": self.text_cache
        }
        joblib.dump(knowledge_data, filename)

    @classmethod
    def load(cls, filename: str, device: str = None,
             clip_model: str = 'ViT-B/32', 
             sam_model_type: str = "vit_h",
             segmentation_method: str = "SLIC",
             similarity_threshold: float = 0.3,
             segmentation_params: dict = None):
        """
        从文件中加载 Knowledge_Base
        """
        data = joblib.load(filename)
        self = cls(
            device=device,
            clip_model=clip_model,
            sam_model_type=sam_model_type,
            segmentation_method=segmentation_method,
            similarity_threshold=similarity_threshold,
            segmentation_params=segmentation_params
        )
        # 重载数据
        self.knowledge = defaultdict(lambda: {
            "text_embeds": None,
            "image_embeds": defaultdict(list)
        }, data["knowledge"])
        self.text_cache = data["text_cache"]
        return self

    def get_class_data(self, class_name: str) -> Dict:
        """
        获取指定类别的完整数据，包括：
        {
          "text_embeds": ...,
          "image_embeds": {attr_name: [...], ...}
        }
        """
        if class_name not in self.text_cache:
            return {}
        return {
            "text_embeds": self.text_cache[class_name],
            "image_embeds": dict(self.knowledge[class_name]["image_embeds"])
        }

    def print_structure(self):
        """
        打印知识库结构概览
        """
        print("="*60 + "\n[Knowledge Base Structure]\n" + "="*60)
        # 打印文本缓存
        print("\n[Text Cache]")
        for cls, data in self.text_cache.items():
            print(f"Class: {cls}")
            print(f"  ├─ Overall Embedding: {data['overall'].shape}")
            print(f"  ├─ Class Embedding: {data['class'].shape}")
            print(f"  └─ Attributes ({len(data['attribute_names'])}):")
            for name, emb in zip(data['attribute_names'], data['attributes']):
                print(f"     └─ {name}: {list(emb.shape)}")

        print("\n[Image Knowledge]")
        for cls, data in self.knowledge.items():
            print(f"Class: {cls}")
            total = sum(len(v) for v in data['image_embeds'].values())
            print(f"  └─ Total Image Embeds: {total}")
            for attr, items in data['image_embeds'].items():
                print(f"     └─ {attr}: {len(items)} items")
                if items:
                    sample = items[0]
                    emb_shape = list(sample['embedding'].shape)
                    patch_info = "Yes" if (sample['patch'] is not None) else "No"
                    print(f"        - Example Embedding Shape: {emb_shape}, Has Patch: {patch_info}")

        print("\n[Index Cache]")
        for cls, data in self.index_cache.items():
            print(f"Class: {cls}")
            print(f"  ├─ Embeddings Shape: {data['embeddings'].shape}")
            print(f"  └─ Mapping Count: {len(data['mapping'])}")
        print("="*60)

    def visualize_segmentation(self, 
                               image: Image.Image, 
                               save_path: str = None, 
                               max_cols: int = 8,
                               figsize: tuple = (20, 20),
                               fontsize: int = 8):
        """
        可视化分割结果，用于调试和观察。
        """
        import matplotlib.pyplot as plt

        patches = self._segment_image(image)
        if not patches:
            print("⚠️ No valid patches found.")
            return

        num_patches = len(patches)
        cols = min(num_patches, max_cols)
        rows = (num_patches + cols - 1) // cols

        fig = plt.figure(figsize=figsize)
        plt.suptitle(f"{self.segmentation_method} Segmentation Results", fontsize=12, y=0.95)

        for i, patch in enumerate(patches, 1):
            ax = fig.add_subplot(rows, cols, i)
            ax.imshow(patch)
            ax.set_title(f"Patch {i}", fontsize=fontsize)
            ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Segmentation results saved at: {save_path}")
        else:
            plt.show()
        plt.close()

        import matplotlib.pyplot as plt

    def query_and_visualize(self, 
                            query_patch: Image.Image, 
                            class_name: str, 
                            top_k: int = 5, 
                            save_path: str = None):
        """
        对指定类别 class_name 的知识库进行查询，并可视化“查询 patch + TopK 匹配 patch”。
        
        :param query_patch: 需要查询的单张图像 patch (PIL.Image)
        :param class_name: 在知识库中匹配的类别名称
        :param top_k: 返回和可视化的前K个结果
        :param save_path: 若不为 None，则将可视化结果保存到指定路径；否则直接 plt.show()
        """

        # 1. 编码查询patch
        with torch.no_grad():
            query_tensor = self.preprocess(query_patch).unsqueeze(0).to(self.device)
            query_embed = self.model.encode_image(query_tensor).squeeze(0)
        
        # 2. 在知识库中查询
        results = self.query(query_embed, class_name, top_k=top_k)
        # 如果没有查到或没有匹配项，直接返回
        if not results:
            print(f"[Warning] No results found for class '{class_name}'!")
            return
        
        self.visualize_query(
            query_patch=query_patch,
            top_k=5,
            results=results,
            class_name=class_name,
            save_path=save_path
        )
    
    def visualize_query(self, query_patch, top_k, results, class_name, save_path):
        #创建画布：1 行 (top_k+1) 列
        num_cols = top_k + 1  # 第一个为 query patch，其余为匹配结果
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

        axes[0].imshow(query_patch)
        axes[0].set_title("Query Patch", fontsize=10)
        axes[0].axis('off')
        
        # 绘制 top_k 匹配到的图像 patch
        for i, (attr_name, similarity, idx_in_attr) in enumerate(results, start=1):
            # 从知识库中找出对应的 patch
            item = self.knowledge[class_name]["image_embeds"][attr_name][idx_in_attr]
            matched_patch = item.get("patch", None)
            
            # 如果存储时并未保存patch或为空，跳过
            if matched_patch is None:
                axes[i].set_title(f"{attr_name}\nNoPatch\nsim:{similarity:.2f}", fontsize=8)
                axes[i].axis('off')
                continue
            
            axes[i].imshow(matched_patch)
            axes[i].set_title(f"{attr_name} sim: {similarity:.2f}", fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Query visualization saved at: {save_path}")
        else:
            plt.show()
            plt.close()

if __name__ == "__main__":
    # 可选参数： segmentation_method in ["SAM", "SLIC", "FELZENSZWALB"]
    kb = Knowledge_Base(
        device="cuda", 
        clip_model="ViT-B/32", 
        sam_model_type="vit_h",
        segmentation_method="SLIC", 
        similarity_threshold=0.2,
        segmentation_params=None  # 可覆盖默认分割参数
    )
    # 2) 准备文本属性
    #    例如: "bicycle: wheels, a frame, handlebars, a seat, and pedals."
    horse_attributes = "horse: a muscular build, a long mane, a flowing tail, slender legs, a large head, expressive eyes, pointed ears, hard hooves."
    kb.process_attributes(horse_attributes)

    bird_attributes = "bird: a light skeleton, feathers covering the body, a beak, two wings, hollow bones, a streamlined body shape, webbed feet."
    kb.process_attributes(bird_attributes)

    bicycle_attributes = "bicycle: wheel, a frame, handlebars, a seat, pedals."
    kb.process_attributes(bicycle_attributes)
    img = Image.open("/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/001420.jpg").convert("RGB")
    xml_path = '/data2/lihl/data/VOCdevkit/VOC2007/Annotations/001420.xml'
    # 对图像做分割（根据 segmentation_method），并与属性文本进行匹配
    results = kb.process_image(img, class_name="horse", xml_file=xml_path)
    # 将结果加入知识库
    kb.add_to_knowledge("horse", results)
    save_directory = "/data/lihl/fsod/model/knowledge_base/debug_sam/patches"
    kb.visualize_and_save_slic_multiscale(img, class_name="horse", save_dir=save_directory, xml_file=xml_path, top_k=5)

    # ...
    #kb2 = Knowledge_Base.load("my_knowledge.pkl", device="cpu")
    # 例如想用另一张图像的某块 patch 做查询
    
    test_img = Image.open("/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/001420.jpg").convert("RGB")
    patches = kb._segment_image(test_img)  # 取出所有过分割块
    if patches:
        query_embed = kb._encode_images([patches[1]]).squeeze(0)
        query_results = kb.query(query_embed, "horse", top_k=5)
        for attr_name, sim, idx_in_attr in query_results:
            print(attr_name, sim, idx_in_attr)

    if len(patches) > 0:
        query_patch = patches[1]
        kb.query_and_visualize(
            query_patch=query_patch,
            class_name="horse",
            top_k=5,
            save_path="/data/lihl/fsod/model/knowledge_base/debug_sam/query_result.png"  # 或者 None
        )
    kb.print_structure()
    kb.visualize_segmentation(img, save_path="/data/lihl/fsod/model/knowledge_base/debug_sam/seg_vis.png", max_cols=5, figsize=(15,15))
    
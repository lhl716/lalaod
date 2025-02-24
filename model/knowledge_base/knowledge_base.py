import torch
import clip
import os
import cv2
from PIL import Image
import numpy as np
import joblib
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
from urllib.request import urlretrieve
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class Knowledge_Base:
    def __init__(self, device: str = None, clip_model: str = 'ViT-B/32', sam_model_type: str = "vit_h"):
        self.sam_device = "cpu"  
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(clip_model, self.device)
        self.knowledge = defaultdict(lambda: {
            "text_embeds": None,
            "image_embeds": defaultdict(list)
        })
        self.text_cache = {}
        self.index_cache = {}
        
        self.similarity_threshold = 0.75
        self.sam = self._init_sam(sam_model_type)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.92,
            crop_n_layers=2,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=50,
            box_nms_thresh=0.7
        )
        self.similarity_thresholds = defaultdict(float)
        self.min_similarity = 0.65  # 全局最低阈值

    
    def _init_sam(self, model_type: str):
        """自动检测并下载SAM权重的初始化方法"""
        # 模型配置
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

        # 验证模型类型
        if model_type not in model_config:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from {list(model_config.keys())}")

        cfg = model_config[model_type]
        os.makedirs(os.path.dirname(cfg["path"]), exist_ok=True)

        # 下载检查（带MD5校验）
        if not self._check_file(cfg["path"], cfg["md5"]):
            print(f"Downloading {model_type} model...")
            self._download_with_progress(cfg["url"], cfg["path"])
            
            if not self._check_file(cfg["path"], cfg["md5"]):
                raise RuntimeError(f"Failed to download valid {model_type} model. "
                                  "Please download manually from: {cfg['url']}")

        # 加载模型
        sam = sam_model_registry[model_type](checkpoint=cfg["path"])
        sam.to(device=self.sam_device)  # 关键修改
        return sam

    def _check_file(self, path: str, expected_md5: str) -> bool:
        """检查文件完整性和MD5"""
        if not os.path.exists(path):
            return False
        
        try:
            import hashlib
            with open(path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            return md5 == expected_md5
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
        
    def _parse_attribute_text(self, text: str) -> Tuple[str, List[str]]:
        """解析属性文本为类名和属性列表"""
        class_name, attributes = text.split(":", 1)
        class_name = class_name.strip()
        attributes = [class_name] + [attr.strip() for attr in attributes.strip(" .").split(", ")]
        return class_name, attributes

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """批量文本编码"""
        with torch.no_grad():
            text_inputs = clip.tokenize(texts).to(self.device)
            return self.model.encode_text(text_inputs)
    
    def process_attributes(self, attribute_text: str) -> Dict[str, torch.Tensor]:
        """处理属性文本生成嵌入（修正版）"""
        class_name, components = self._parse_attribute_text(attribute_text)
        embeddings = self._encode_texts(components)
        
        # 分解为独立的张量
        self.text_cache[class_name] = {
            "overall": embeddings[0],
            "class": embeddings[1],
            "attributes": [embeddings[i] for i in range(2, len(components))],  # 改为列表存储
            "attribute_names": components[2:]
        }
        return self.text_cache[class_name]
    
    def _segment_image(self, image: Image.Image) -> List[Image.Image]:
        """优化后的小物体分割策略"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 转换数据类型确保兼容SAM CPU处理
        cv_image = cv_image.astype(np.uint8)
        
        masks = self.mask_generator.generate(cv_image)
        
        # 优化过滤策略：按稳定性+面积综合排序
        valid_masks = sorted(
            [m for m in masks if m['stability_score'] > 0.9],
            key=lambda x: x['stability_score'] * np.sqrt(x['area']),
            reverse=True
        )[:12]  # 最多保留12个最显著区域

        # 非极大值抑制(NMS)处理重叠区域
        boxes = [m["bbox"] for m in valid_masks]
        scores = [m["predicted_iou"] for m in valid_masks]
        indices = self._nms(np.array(boxes), np.array(scores), 0.4)
        
        return [self._crop_mask(image, valid_masks[i]) for i in indices]

    def _nms(self, boxes, scores, threshold):
        """非极大值抑制实现"""
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = x1 + boxes[:,2]
        y2 = y1 + boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        return keep

    def _crop_mask(self, image, mask):
        """精确裁剪掩码区域"""
        mask_array = mask["segmentation"]
        y, x = np.where(mask_array)
        if len(x) == 0 or len(y) == 0:
            return None
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return image.crop((x_min, y_min, x_max+1, y_max+1))

    
    def _encode_images(self, patches: List[Image.Image]) -> torch.Tensor:
        """批量图像编码"""
        if not all(isinstance(p, Image.Image) for p in patches):
            raise ValueError("All patches must be PIL.Image instances")
        with torch.no_grad():
            image_inputs = torch.stack([self.preprocess(patch) for patch in patches]).to(self.device)
            return self.model.encode_image(image_inputs)
    
    def _find_best_matches(self, 
                         image_embeds: torch.Tensor, 
                         text_embeds: torch.Tensor) -> List[int]:
        """使用矩阵运算高效匹配最佳属性（修正版）"""
        # 转换为CLIP的输入格式
        logit_scale = self.model.logit_scale.exp()  # 获取CLIP的温度参数
        # 添加维度校验
        if image_embeds.dim() == 1:
            image_embeds = image_embeds.unsqueeze(0)
        if text_embeds.dim() == 1:
            text_embeds = text_embeds.unsqueeze(0)
            
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # 计算logits（CLIP原始相似度计算）
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        print(torch.argmax(logits_per_image, dim=-1).cpu().numpy())
        return torch.argmax(logits_per_image, dim=-1).cpu().numpy()
    
    def process_image(self, image: Image.Image, class_name: str, save_patches: bool = False) -> Dict:
        """处理单张图像"""
        if class_name not in self.text_cache:
            raise ValueError(f"Class {class_name} not initialized")
        
        # 分割图像
        patches = self._segment_image(image)
        if not patches:
            return {}
        
        # 编码图像块
        image_inputs = torch.stack([self.preprocess(patch) for patch in patches]).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.encode_image(image_inputs)
        
        # 获取文本属性
        text_data = self.text_cache[class_name]
        attr_embeds = torch.stack(text_data["attributes"]).to(self.device)
        
        # 计算相似度
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        attr_embeds_norm = attr_embeds / attr_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds_norm @ attr_embeds_norm.T) * self.model.logit_scale.exp()
        
        # 匹配结果
        results = defaultdict(list)
        for i in range(len(patches)):
            max_sim, max_idx = torch.max(similarity[i], dim=0)
            max_idx = max_idx.item()  # 关键修复：转换为Python整数
            
            if max_sim < self.similarity_threshold:
                continue
                
            attr_name = text_data["attribute_names"][max_idx]
            results[attr_name].append({
                "embedding": image_embeds[i].cpu().clone(),
                "similarity": max_sim.item(),
                "patch": patches[i] if save_patches else None
            })
            
        # 在process_image末尾添加诊断输出
        print(f"[Diagnostic] Class: {class_name}")
        print(f"| Patches: {len(patches)} | Matches: {sum(len(v) for v in results.values())}")
        print(f"| Similarity Range: {similarity.min():.2f}-{similarity.max():.2f}")
        
        return results
    
    def add_to_knowledge(self, class_name: str, image_results: Dict):
        """修正后的添加方法，存储完整字典结构"""
        if class_name not in self.text_cache:
            raise ValueError(f"Class {class_name} not initialized")
        
        for attr_name, items in image_results.items():
            # 这里改为存储完整字典而非仅embedding
            self.knowledge[class_name]["image_embeds"][attr_name].extend(items)
        
        # 更新索引缓存
        if class_name in self.index_cache:
            del self.index_cache[class_name]
    
    def _build_index(self, class_name: str):
        """修正后的索引构建方法"""
        all_embeds = []
        mapping = []
        
        for attr, items in self.knowledge[class_name]["image_embeds"].items():
            for idx, item in enumerate(items):
                # 确保嵌入张量移动到CPU后再转换为numpy
                all_embeds.append(item["embedding"].cpu().numpy())
                mapping.append((attr, idx))
        
        self.index_cache[class_name] = {
            "embeddings": np.stack(all_embeds),
            "mapping": mapping
        }
    
    def query(self, 
            query_embed: torch.Tensor, 
            class_name: str, 
            top_k: int = 5) -> List[Tuple[str, float, int]]:
        """修正后的查询方法"""
        if class_name not in self.index_cache:
            self._build_index(class_name)
        
        index_data = self.index_cache[class_name]
        # 确保查询嵌入移至CPU
        query_np = query_embed.cpu().numpy()
        
        # 使用余弦相似度
        norms = np.linalg.norm(index_data["embeddings"], axis=1)
        query_norm = np.linalg.norm(query_np)
        similarities = (index_data["embeddings"] @ query_np) / (norms * query_norm)
        
        top_indices = np.argsort(-similarities)[:top_k]
        
        return [(
            index_data["mapping"][i][0],  # 属性名
            float(similarities[i]),       # 相似度
            index_data["mapping"][i][1]   # 原始索引
        ) for i in top_indices]
    
    def save(self, filename: str):
        """保存知识库"""
        knowledge_data = {
            "knowledge": dict(self.knowledge),
            "text_cache": self.text_cache
        }
        joblib.dump(knowledge_data, filename)
    
    @classmethod
    def load(cls, filename: str, device: str = None):
        """加载知识库"""
        self = cls(device=device)
        data = joblib.load(filename)
        self.knowledge = defaultdict(lambda: {
            "text_embeds": None,
            "image_embeds": defaultdict(list)
        }, data["knowledge"])
        self.text_cache = data["text_cache"]
        return self

    def get_class_data(self, class_name: str) -> Dict:
        """获取指定类别的完整数据"""
        return {
            "text_embeds": self.text_cache[class_name],
            "image_embeds": dict(self.knowledge[class_name]["image_embeds"])
        }
    
    def _wrap_text(self, text: str, max_length: int = 25) -> str:
        """自动换行文本处理"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_length:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def print_structure(self):
        """打印知识库完整结构"""
        print("="*60 + "\nKnowledge Base Structure\n" + "="*60)
        
        # 打印文本缓存
        print("\n[Text Cache]")
        for cls, data in self.text_cache.items():
            print(f"Class: {cls}")
            print(f"├─ Overall Embedding: {data['overall'].shape}")
            print(f"├─ Class Embedding: {data['class'].shape}")
            print(f"└─ Attributes ({len(data['attribute_names'])}):")
            for name, emb in zip(data['attribute_names'], data['attributes']):
                print(f"   ├─ Embedding of {name}: {emb.shape}")

        # 打印图像知识
        print("\n[Image Knowledge]")
        for cls, data in self.knowledge.items():
            print(f"Class: {cls}")
            total = sum(len(v) for v in data['image_embeds'].values())
            print(f"└─ Total Image Embeds: {total}")
            for attr, items in data['image_embeds'].items():
                print(f"   ├─ {attr} ({len(items)} items)")
                if items:
                    sample = items[0]
                    print(f"   │  ├─ Embedding: {sample['embedding'].shape}")
                    print(f"   │  ├─ Image Patch: {sample['patch'].size}")
                    print(f"   │  └─ Has Patch: {'patch' in sample}")

        # 打印索引缓存
        print("\n[Index Cache]")
        for cls, data in self.index_cache.items():
            print(f"Class: {cls}")
            print(f"├─ Embeddings Shape: {data['embeddings'].shape}")
            print(f"└─ Mapping Count: {len(data['mapping'])}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 创建保存目录
    save_dir = "/data/lihl/fsod/model/knowledge_base/result"
    os.makedirs(save_dir, exist_ok=True)
    # 初始化知识库
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kb = Knowledge_Base(sam_model_type="vit_h")

    # 示例1：处理马类的属性文本
    horse_attributes = "Horses: a muscular build, a long mane, a flowing tail, slender legs, a large head, expressive eyes, pointed ears, and hard hooves."
    kb.process_attributes(horse_attributes)

    # 示例2：处理鸟类的属性文本
    bird_attributes = "Birds: a light skeleton, feathers covering the body, a beak, two wings, hollow bones, a streamlined body shape, and webbed feet."
    kb.process_attributes(bird_attributes)

    # 处理马类图像示例
    def process_horse_image(image_path):
        img = Image.open(image_path).convert("RGB")
        results = kb.process_image(img, "Horses", save_patches=True)
        
        # 可视化分割结果
        patch_info = []
        for attr, items in results.items():
            for item in items:
                if item["patch"] is not None:
                    patch_info.append( (item["patch"], attr) )

        fig = plt.figure(figsize=(16, 8))
        plt.suptitle("SAM Segmented Patches with Attributes", fontsize=14, y=0.95)
        
        cols = 4
        rows = (len(patch_info) + cols - 1) // cols
        
        for i, (patch, attr) in enumerate(patch_info, 1):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(patch)
            ax.axis('off')
            wrapped_text = kb._wrap_text(attr)
            ax.set_title(wrapped_text, fontsize=9, pad=6, color='#2c3e50')
        
        plt.tight_layout(pad=2.0)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"segments_{timestamp}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 分割结果已保存至: {save_path}")
        
        # 存入知识库
        kb.add_to_knowledge("Horses", results)
        print(f"📥 已添加 {sum(len(v) for v in results.values())} 个特征到知识库")
            

    # 查询示例
    def query_example(query_path):
        
        query_img = Image.open(query_path).convert("RGB")
        patches = kb._segment_image(query_img)
        
        if not patches:
            print("⚠️ 未检测到有效区域")
            return
            
        # 使用第一个patch查询
        query_embed = kb._encode_images([patches[0]]).squeeze(0)
        results = kb.query(query_embed, "Horses", top_k=3)
        
        # 可视化结果
        fig = plt.figure(figsize=(15, 5))
        
        # 查询图像
        plt.subplot(1,4,1)
        plt.title("查询区域", fontproperties="SimHei")
        plt.imshow(patches[0])
        plt.axis('off')
        
        # 匹配结果
        for i, (attr, sim, idx) in enumerate(results):
            plt.subplot(1,4,i+2)
            item = kb.knowledge["Horses"]["image_embeds"][attr][idx]
            plt.imshow(item["patch"])
            plt.title(f"{attr}\n相似度: {sim:.2f}", fontproperties="SimHei")
            plt.axis('off')
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"query_{timestamp}.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"✅ 查询结果已保存至: {save_path}")
            

    # 执行流程 -------------------------------------------------
    # 处理训练图像
    train_images = [
        "/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000214.jpg",
        "/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000328.jpg"
    ]
    
    for img_path in train_images:
        print(f"\n🔍 正在处理: {img_path}")
        process_horse_image(img_path)
    
    # 保存知识库
    kb.save("/data/lihl/fsod/model/knowledge_base/animal_knowledge.pkl")
    print("\n💾 知识库保存完成")
    
    # 执行查询
    query_path = "/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000332.jpg"
    print(f"\n🔎 正在查询: {query_path}")
    query_example(query_path)
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(clip_model, self.device)
        self.knowledge = defaultdict(lambda: {
            "text_embeds": None,
            "image_embeds": defaultdict(list)
        })
        self.text_cache = {}
        self.index_cache = {}
        
        self.sam = self._init_sam(sam_model_type)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=64,
            pred_iou_thresh=0.92,
            stability_score_thresh=0.95,
            crop_n_layers=2,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100  # 过滤小区域
        )
        self.similarity_threshold = 0.2
    
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
        sam.to(device=self.device)
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
        """使用SAM进行高质量图像分割"""
        # 转换图像格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_size = cv_image.shape[:2]
        
        # 调整图像尺寸（保持长边不超过1024）
        scale_factor = max(1024 / max(original_size), 1.0)
        new_size = (int(original_size[1] * scale_factor), 
                   int(original_size[0] * scale_factor))
        resized_image = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # 生成掩码
        with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu"):
            masks = self.mask_generator.generate(resized_image)
        
        # 过滤和排序掩码
        valid_masks = [
            mask for mask in sorted(masks, key=(lambda x: x['area']), reverse=True) 
            if mask['area'] > 500 and mask['predicted_iou'] > 0.9
        ]  # 最多取8个最重要的区域
        
        # 提取分割区域
        patches = []
        for mask in valid_masks:
            # 将掩码缩放到原始尺寸
            mask_array = cv2.resize(
                mask["segmentation"].astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 提取掩码区域
            masked = cv2.bitwise_and(
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                mask=mask_array
            )
            
            # 转换为PIL Image并裁剪有效区域
            y, x = np.where(mask_array)
            if len(x) == 0 or len(y) == 0:
                continue
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            
            patch = Image.fromarray(cv2.cvtColor(masked[y_min:y_max+1, x_min:x_max+1], cv2.COLOR_BGR2RGB))
            patches.append(patch)
        
        return patches

    
    def _encode_images(self, patches: List[Image.Image]) -> torch.Tensor:
        """批量图像编码"""
        if not all(isinstance(p, Image.Image) for p in patches):
            raise ValueError("All patches must be PIL.Image instances")
        with torch.no_grad():
            image_inputs = torch.stack([self.preprocess(patch) for patch in patches]).to(self.device)
            return self.model.encode_image(image_inputs)
    
    def _find_best_matches(
        self, 
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:  # 修改返回值类型
        """
        返回：
        - matches: 匹配的索引数组（无效匹配标记为 -1）
        - similarity_matrix: 完整的相似度矩阵
        """
        # 归一化嵌入向量
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵（未缩放）
        similarity_matrix = image_embeds @ text_embeds.T
        
        # 应用 CLIP 的温度缩放
        logit_scale = self.model.logit_scale.exp()
        scaled_similarity = similarity_matrix * logit_scale
        
        # 匹配逻辑（基于缩放后的相似度）
        max_sim = scaled_similarity.max(dim=1).values
        valid_mask = max_sim > self.similarity_threshold
        
        # 生成匹配结果
        matches = torch.full(
            (scaled_similarity.size(0),), 
            -1, 
            dtype=torch.long, 
            device=image_embeds.device
        )
        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() > 0:
            matches[valid_indices] = torch.argmax(scaled_similarity[valid_indices], dim=1)
        
        # 返回匹配索引和原始相似度矩阵（未缩放）
        return matches.cpu().numpy(), similarity_matrix.cpu()
    
    def process_image(
        self, 
        image: Image.Image, 
        class_name: str, 
        save_patches: bool = False
    ) -> Dict:
        """处理单张图像并应用阈值过滤"""
        if class_name not in self.text_cache:
            raise ValueError(f"Class {class_name} not processed")
        
        # 图像分割
        patches = self._segment_image(image)
        if not patches:
            return {}
        
        # 编码图像块
        image_embeds = self._encode_images(patches)
        
        # 获取文本属性嵌入
        text_data = self.text_cache[class_name]
        if not text_data["attributes"]:
            return {}
        
        attribute_tensor = torch.stack(text_data["attributes"]).to(self.device)
        
        # 获取匹配结果和相似度矩阵
        matches, similarity_matrix = self._find_best_matches(image_embeds, attribute_tensor)
        
        # 构建结果（应用阈值过滤）
        results = defaultdict(list)
        for idx, match_idx in enumerate(matches):
            if match_idx == -1:  # 跳过无效匹配
                continue
            
            attr_name = text_data["attribute_names"][match_idx]
            similarity = similarity_matrix[idx, match_idx].item()  # 使用原始相似度
            
            if similarity >= self.similarity_threshold:
                results[attr_name].append({
                    "embedding": image_embeds[idx].cpu().clone(),
                    "patch": patches[idx] if save_patches else None,
                    "similarity": similarity  # 存储相似度值
                })
        
        # 按相似度降序排序
        for attr in results:
            results[attr].sort(key=lambda x: x["similarity"], reverse=True)
        
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
                all_embeds.append(item["embedding"].numpy())
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
        # 加载并处理图像
        img = Image.open(image_path).convert("RGB")
        results = kb.process_image(img, "Horses", save_patches=True)
    
        patch_info = []
        for attr_name, items in results.items():
            for item in items:
                if item["patch"] is not None:
                    patch_info.append( (item["patch"], attr_name) )
        
        # 可视化分割结果（带属性名称标签）
        fig = plt.figure(figsize=(16, 8))
        plt.suptitle("SAM Segmented Patches with Attributes", fontsize=14, y=0.95)
        
        cols = 4
        rows = (len(patch_info) + cols - 1) // cols
        
        for i, (patch, attr) in enumerate(patch_info, 1):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(patch)
            ax.axis('off')
            
            # 自动换行处理长文本
            wrapped_text = kb._wrap_text(attr, max_length=25)
            ax.set_title(wrapped_text, 
                        fontsize=9, 
                        pad=6,
                        color='#2c3e50',
                        fontweight='semibold',
                        loc='center',
                        wrap=True)
        
        plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
        
        # 保存分割结果
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join("/data/lihl/fsod/model/knowledge_base/result", 
                                f"sam_segments_{timestamp}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved SAM segments to {save_path}")
        
        # 添加到知识库
        kb.add_to_knowledge("Horses", results)
        added_count = sum(len(v) for v in results.values())
        print(f"📥 Added {added_count} embeddings to Horses knowledge")

    # 处理多个示例图像
    process_horse_image("/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000214.jpg")  # 替换为实际图片路径
    process_horse_image("/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000328.jpg")

    # 保存知识库
    kb.save("/data/lihl/fsod/model/knowledge_base/animal_knowledge.pkl")
    
    # 加载知识库（可选）
    # kb = Knowledge_Base.load("animal_knowledge.pkl", device=device)

    def query_example(query_image_path):
        # 加载并分割查询图像
        query_img = Image.open(query_image_path).convert("RGB")
        patches = kb._segment_image(query_img)
        print(len(patches))
        print(patches.shape)
        for index in range(len(patches)):
            # 选择第一个patch作为查询示例
            query_embed = kb._encode_images([patches[index]]).squeeze(0)
            
            # 在知识库中搜索
            results = kb.query(query_embed, "Horses", top_k=3)
            
            # 可视化查询结果
            fig = plt.figure(figsize=(15, 5))
            plt.subplot(1,4,1)
            plt.title("Query Patch")
            plt.imshow(patches[0])
            plt.axis('off')
            
            for i, (attr_name, similarity, idx) in enumerate(results):
                plt.subplot(1,4,i+2)
                plt.title(f"{attr_name}\nSimilarity: {similarity:.2f}")
                # 修正访问方式
                example_item = kb.knowledge["Horses"]["image_embeds"][attr_name][idx]
                if example_item.get("patch"):
                    plt.imshow(example_item["patch"])
                plt.axis('off')
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(save_dir, f"query_{index}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved query result to {save_path}")
    kb.print_structure()

    # 执行查询（使用新图像）
    query_example("/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000332.jpg")  # 替换为实际查询图片路径
import os
import cv2
import json
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict
from skimage.segmentation import slic
from transformers import CLIPProcessor, CLIPModel

############################################################
# 1. VOC相关辅助函数：解析SegmentationClass，提取指定类区域
############################################################

# 标准 VOC2012 类别映射(0-20)，若使用VOC2007保证对应关系一致
# 背景:0, aeroplane:1, bicycle:2, bird:3, boat:4, bottle:5, bus:6,
# car:7, cat:8, chair:9, cow:10, diningtable:11, dog:12, horse:13,
# motorbike:14, person:15, pottedplant:16, sheep:17, sofa:18,
# train:19, tv/monitor:20.
voc_class_map = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}


def extract_voc_segments(
    img_path: str,
    seg_path: str,
    class_name: str,
    save_dir: str = None
) -> List[Image.Image]:
    """
    从VOC数据集中加载img和其对应的语义分割图(seg)，
    找到 class_name 对应的所有连通区域，并输出每个区域对应的独立子图(仅该类像素可见，其他区域置黑)。
    
    :param img_path: 原始图像路径
    :param seg_path: 分割mask路径（index图或标准VOC调色板图）
    :param class_name: 目标类名称，在 voc_class_map 中寻找
    :param save_dir: 若指定，则会将提取到的子图(蒙版处理后)存到该目录
    :return: 子图的列表，每个子图为包含目标类其中一个连通域的 PIL.Image（RGB形式）。
    """
    if class_name not in voc_class_map:
        raise ValueError(f"类 {class_name} 不在 voc_class_map 中，请检查映射！")

    class_idx = voc_class_map[class_name]

    # 读取RGB图像
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 读取分割mask（应为单通道 index图 or 调色板图）
    seg_img = Image.open(seg_path)
    seg_img = np.array(seg_img)  # 转成numpy数组

    # 若 seg_img 形状与原图对应则 seg_img.shape == (h, w)
    # 否则需要先检查或resize
    if seg_img.shape[0] != h or seg_img.shape[1] != w:
        print("[Warn] Segmentation shape != image shape, resizing mask to image shape.")
        seg_img = cv2.resize(seg_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # 找到属于 class_idx 的像素
    print(f'class name:{class_name}, class idx:{class_idx}')
        # 统计不同数字的个数
    unique, counts = np.unique(seg_img, return_counts=True)

    # 打印结果
    for u, c in zip(unique, counts):
        print(f"数字 {u}: {c} 个")
    target_mask = (seg_img == class_idx).astype(np.uint8)

    # 连通域标记
    num_labels, comp_map = cv2.connectedComponents(target_mask, connectivity=8)
    # comp_map 中的取值范围为 [0, num_labels-1], 其中0对应背景

    extracted_images = []
    for label_val in range(1, num_labels):
        comp_mask = (comp_map == label_val).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        if len(ys) == 0:
            continue
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        # 将非连通域像素置黑
        roi = np.zeros_like(img)
        roi[comp_mask == 1] = img[comp_mask == 1]

        # 截取最小包围盒
        cropped = roi[ymin:ymax+1, xmin:xmax+1, :]

        # 转为 PIL
        pil_cropped = Image.fromarray(cropped, mode="RGB")
        extracted_images.append(pil_cropped)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{class_name}_comp_{label_val}.jpg")
            pil_cropped.save(out_path)
            print(f"Saved {class_name} region -> {out_path}")

    return extracted_images


############################################################
# 2. Knowledge_Base 类：使用 HuggingFace CLIP + SLIC
############################################################

class Knowledge_Base:
    def __init__(
        self,
        device: str = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        similarity_threshold: float = 0.3,
        segmentation_params: dict = None
    ):
        """
        使用 HuggingFace CLIP + SLIC 过分割 的知识库类。
        
        :param device: 指定计算设备（如 "cuda" 或 "cpu"），默认自动检测。
        :param clip_model_name: 预训练CLIP模型名称，默认 "openai/clip-vit-base-patch32"。
        :param similarity_threshold: 在属性匹配时使用的相似度阈值。
        :param segmentation_params: SLIC分割参数，如 {"n_segments":200, "compactness":10, "sigma":1}。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        
        # 初始化HF CLIP模型和处理器
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # 知识库存储结构
        self.knowledge = defaultdict(lambda: {
            "text_embeds": None,
            "image_embeds": defaultdict(list)
        })
        
        # 用于缓存属性文本的嵌入
        self.text_cache = {}
        
        # 检索索引缓存
        self.index_cache = {}

        # SLIC 默认参数
        self.default_seg_params = {
            "n_segments": 200,
            "compactness": 10.0,
            "sigma": 1.0
        }
        self.seg_params = {**self.default_seg_params, **(segmentation_params or {})}
    
    def update_segmentation_params(self, new_params: dict):
        """
        动态更新SLIC分割参数。
        """
        self.seg_params = {**self.seg_params, **new_params}
        print(f"✅ 分割参数已更新: {self.seg_params}")

    def _parse_attribute_text(self, text: str) -> Tuple[str, List[str]]:
        """
        将字符串解析成 (类名, [整体描述, 类名, 属性1, 属性2, ...]) 的格式。
        例如 "Bicycle: wheels, a frame, handlebars" 解析为
        ("Bicycle", ["Bicycle", "Bicycle", "wheels", "a frame", "handlebars"]).
        """
        class_name, attributes = text.split(":", 1)
        class_name = class_name.strip()
        components = [class_name] + [attr.strip() for attr in attributes.strip(" .").split(", ")]
        return class_name, components

    def _encode_texts_hf(self, texts: List[str]) -> torch.Tensor:
        """
        调用 HuggingFace CLIP 的模型进行文本特征提取，返回 (N, D) 的张量。
        """
        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        with torch.no_grad():
            text_embeds = self.clip_model.get_text_features(**inputs)  # [N, D]
        return text_embeds
    
    def process_attributes(self, attribute_text: str) -> Dict[str, torch.Tensor]:
        """
        处理属性文本，并缓存相应的文本嵌入。
        返回结构:
        {
          "overall": tensor(D),
          "class": tensor(D),
          "attributes": [tensor(D), ...],
          "attribute_names": [str, ...]
        }
        """
        class_name, components = self._parse_attribute_text(attribute_text)
        text_embeds = self._encode_texts_hf(components)

        self.text_cache[class_name] = {
            "overall": text_embeds[0],
            "class": text_embeds[1],
            "attributes": text_embeds[2:],
            "attribute_names": components[2:]
        }
        return self.text_cache[class_name]

    def _encode_images_hf(self, patches: List[Image.Image]) -> torch.Tensor:
        """
        调用 HuggingFace CLIP 进行图像特征提取，返回 (N, D) 的张量。
        """
        if not patches:
            return torch.empty(0)
        
        inputs = self.clip_processor(
            images=patches,
            return_tensors="pt",
            padding=True
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        
        with torch.no_grad():
            image_embeds = self.clip_model.get_image_features(**inputs)
        return image_embeds
    
    def _segment_image_slic(self, pil_img: Image.Image) -> List[Image.Image]:
        """
        对给定PIL图像执行SLIC过分割，返回若干patch（PIL Image）。
        并生成一个可视化的整合图片(仅过分割展示)，可在其他函数中保存。
        """
        np_img = np.array(pil_img.convert("RGB"))
        h, w = np_img.shape[:2]
        
        segments = slic(
            np_img,
            n_segments=self.seg_params["n_segments"],
            compactness=self.seg_params["compactness"],
            sigma=self.seg_params["sigma"]
        )
        
        unique_vals = np.unique(segments)
        seg_info = []
        for val in unique_vals:
            mask = (segments == val)
            area = np.count_nonzero(mask)
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                continue
            seg_info.append((val, area, xs.min(), xs.max(), ys.min(), ys.max()))
        
        seg_info.sort(key=lambda x: x[1], reverse=True)
        top_k = 10  # 避免patch过多，保留面积最大的若干块
        seg_info = seg_info[:top_k]

        patches = []
        # 提取patch
        for (val, area, xmin, xmax, ymin, ymax) in seg_info:
            cropped = np_img[ymin:ymax+1, xmin:xmax+1, :]
            patch_img = Image.fromarray(cropped, "RGB")
            patches.append(patch_img)
        
        return patches, segments  # segments 也返回，用于可视化着色
    
    def _save_slic_visualization(
        self, 
        pil_img: Image.Image, 
        segments: np.ndarray, 
        save_path: str, 
        title: str = "SLIC Over-segmentation"
    ):
        """
        将SLIC分割区域以随机颜色叠加到原图上，并保存结果。
        """
        np_img = np.array(pil_img.convert("RGB"))
        out_img = np_img.copy()
        h, w = np_img.shape[:2]

        # 随机颜色表
        unique_vals = np.unique(segments)
        color_map = {}
        for val in unique_vals:
            color_map[val] = np.random.randint(0, 255, (3,), dtype=np.uint8)
        
        # 给每个 segment 着色
        for y in range(h):
            for x in range(w):
                val = segments[y, x]
                out_img[y, x] = color_map[val] * 0.5 + out_img[y, x] * 0.5  # 叠加
        
        plt.figure(figsize=(8, 6))
        plt.imshow(out_img)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[SLIC Vis] Saved -> {save_path}")

    def _compute_cosine_sim(self, img_embeds: torch.Tensor, txt_embeds: torch.Tensor) -> torch.Tensor:
        """
        计算图像特征和文本特征间的余弦相似度，返回 (num_imgs, num_texts)。
        """
        if img_embeds.shape[0] == 0 or txt_embeds.shape[0] == 0:
            return torch.empty((0, 0))
        
        img_norm = img_embeds / (img_embeds.norm(dim=1, keepdim=True) + 1e-8)
        txt_norm = txt_embeds / (txt_embeds.norm(dim=1, keepdim=True) + 1e-8)
        return img_norm @ txt_norm.t()

    def _find_best_matches(
        self, 
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        对 image_embeds 中每个元素找到与 text_embeds 最相似的索引。
        无效匹配用 -1 标记。
        """
        if image_embeds.shape[0] == 0 or text_embeds.shape[0] == 0:
            return np.array([], dtype=np.int64), torch.empty((0, 0))
        
        similarity_matrix = self._compute_cosine_sim(image_embeds, text_embeds)
        max_vals, max_indices = similarity_matrix.max(dim=1)
        
        matches = []
        for i, v in enumerate(max_vals):
            if v.item() >= self.similarity_threshold:
                matches.append(max_indices[i].item())
            else:
                matches.append(-1)
        
        return np.array(matches), similarity_matrix

    def process_cropped_image(
        self,
        pil_img: Image.Image,
        class_name: str,
        save_dir: str,
        prefix: str = ""
    ) -> Dict:
        """
        对一张已裁剪好的图像(只包含特定类的区域)执行:
        1) SLIC过分割(并保存过分割可视化图)
        2) 提取patch并与该类的文本属性匹配
        3) 保存匹配结果可视化图
        
        返回 { attr_name: [ { "embedding":..., "patch":..., "similarity":... }, ... ], ... }
        """
        if class_name not in self.text_cache:
            raise ValueError(f"{class_name} 未在文本属性中注册，请先调用 `process_attributes`。")
        
        os.makedirs(save_dir, exist_ok=True)

        # 1. 过分割 + 可视化保存
        patches, segments = self._segment_image_slic(pil_img)
        slic_vis_path = os.path.join(save_dir, f"{prefix}_slic.png")
        self._save_slic_visualization(pil_img, segments, slic_vis_path, title=f"{prefix} SLIC")

        # 2. 提取图像特征
        image_embeds = self._encode_images_hf(patches)
        
        # 3. 与文本属性匹配
        text_data = self.text_cache[class_name]
        txt_embeds = text_data["attributes"]
        attr_names = text_data["attribute_names"]
        
        if txt_embeds.shape[0] == 0 or image_embeds.shape[0] == 0:
            return {}
        
        matches, sim_matrix = self._find_best_matches(image_embeds, txt_embeds)
        
        results = defaultdict(list)
        for i, attr_idx in enumerate(matches):
            if attr_idx == -1:
                continue
            sim_score = sim_matrix[i, attr_idx].item()
            results[attr_names[attr_idx]].append({
                "embedding": image_embeds[i].cpu().clone(),
                "patch": patches[i],
                "similarity": sim_score
            })
        
        # 排序
        for k in results:
            results[k].sort(key=lambda x: x["similarity"], reverse=True)

        # 4. 保存“图文匹配后的结果”可视化：只展示匹配到的patch
        matched_patches = []
        matched_titles = []
        for attr_name, items in results.items():
            for rec in items:
                matched_patches.append(rec["patch"])
                matched_titles.append(f"{attr_name}\n{rec['similarity']:.2f}")

        if matched_patches:
            self._save_matched_patches_figure(
                matched_patches,
                matched_titles,
                os.path.join(save_dir, f"{prefix}_matched.png"),
                title=f"{prefix} Patches Matched"
            )
        
        return results
    
    def _save_matched_patches_figure(
        self,
        patches: List[Image.Image],
        titles: List[str],
        save_path: str,
        title: str = "Matched Patches"
    ):
        """
        将多个patch合并到一个figure上显示，标题为 属性+相似度。
        """
        cols = 5
        rows = (len(patches) + cols - 1) // cols
        plt.figure(figsize=(4*cols, 4*rows))
        plt.suptitle(title)
        
        for i, (img_pil, t) in enumerate(zip(patches, titles), start=1):
            plt.subplot(rows, cols, i)
            plt.imshow(img_pil)
            plt.title(t, fontsize=9)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Matched Patches] Saved -> {save_path}")
    
    def add_to_knowledge(self, class_name: str, image_results: Dict):
        """
        将若干图像patch对应的属性匹配结果加入知识库。
        """
        if class_name not in self.text_cache:
            raise ValueError(f"{class_name} 未初始化文本属性")
        
        for attr_name, items in image_results.items():
            self.knowledge[class_name]["image_embeds"][attr_name].extend(items)
        
        if class_name in self.index_cache:
            del self.index_cache[class_name]

    def _build_index(self, class_name: str):
        """
        为某个类构建检索索引
        """
        all_embeds = []
        mapping = []
        
        for attr, items in self.knowledge[class_name]["image_embeds"].items():
            for idx, item in enumerate(items):
                all_embeds.append(item["embedding"].numpy())
                mapping.append((attr, idx))
        
        if not all_embeds:
            self.index_cache[class_name] = {
                "embeddings": np.empty((0,)),
                "mapping": []
            }
            return
        
        self.index_cache[class_name] = {
            "embeddings": np.stack(all_embeds, axis=0),
            "mapping": mapping
        }
    
    def query(
        self, 
        query_embed: torch.Tensor, 
        class_name: str, 
        top_k: int = 5,
        save_path: str = None
    ) -> List[Tuple[str, float, int]]:
        """
        在知识库中检索与 query_embed 最相似的 top_k 个属性图像patch。
        返回: [(attr_name, similarity, idx_in_attr_list), ...]
        若提供 save_path，则保存可视化查询结果图片
        """
        if class_name not in self.index_cache:
            self._build_index(class_name)
        
        index_data = self.index_cache[class_name]
        if index_data["embeddings"].size == 0:
            return []
        
        index_embeds = index_data["embeddings"]
        mapping = index_data["mapping"]
        
        q = query_embed.cpu().numpy()
        norms = np.linalg.norm(index_embeds, axis=1)
        norm_q = np.linalg.norm(q)
        sim = (index_embeds @ q) / (norms * norm_q + 1e-8)
        
        top_indices = np.argsort(-sim)[:top_k]
        results = []
        patches_for_fig = []
        titles_for_fig = []
        
        for i in top_indices:
            attr_name, idx_in_attr = mapping[i]
            similarity_val = float(sim[i])
            results.append((attr_name, similarity_val, idx_in_attr))
            
            # 拿到对应patch
            patch_info = self.knowledge[class_name]["image_embeds"][attr_name][idx_in_attr]
            patches_for_fig.append(patch_info["patch"])
            titles_for_fig.append(f"{attr_name}\n{similarity_val:.2f}")
        
        # 如需可视化
        if save_path and patches_for_fig:
            cols = len(patches_for_fig)
            plt.figure(figsize=(4*cols, 4))
            plt.suptitle(f"Query Top-{top_k} for {class_name}")
            
            for i, (pt, title_str) in enumerate(zip(patches_for_fig, titles_for_fig), start=1):
                plt.subplot(1, cols, i)
                plt.imshow(pt)
                plt.title(title_str, fontsize=9)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[Query Results] Saved -> {save_path}")

        return results

    def save(self, filename: str):
        """
        序列化保存知识库（不包括 index_cache）。
        """
        data = {
            "knowledge": dict(self.knowledge),
            "text_cache": self.text_cache
        }
        joblib.dump(data, filename)
    
    @classmethod
    def load(cls, filename: str, device: str = None):
        """
        从文件加载知识库，返回 Knowledge_Base 实例。
        """
        self = cls(device=device)
        loaded = joblib.load(filename)
        
        self.knowledge = defaultdict(
            lambda: {"text_embeds": None, "image_embeds": defaultdict(list)},
            loaded["knowledge"]
        )
        self.text_cache = loaded["text_cache"]
        return self

    def print_structure(self):
        """
        打印知识库的整体结构，查看各类存了哪些嵌入。
        """
        print("\n========== Knowledge Base Structure ==========")
        print("[Text Cache]")
        for cls_name, data in self.text_cache.items():
            print(f"Class: {cls_name}")
            print(f"├─ Overall Embedding Shape: {tuple(data['overall'].shape)}")
            print(f"├─ Class Embedding Shape: {tuple(data['class'].shape)}")
            print(f"└─ #Attributes: {len(data['attributes'])} -> {data['attribute_names']}")
        
        print("\n[Knowledge]")
        for cls_name, data in self.knowledge.items():
            total_emb = sum(len(v) for v in data["image_embeds"].values())
            print(f"Class: {cls_name} | total image embeddings: {total_emb}")
            for attr, items in data["image_embeds"].items():
                print(f"  └─ '{attr}': {len(items)} items.")
        print("==============================================\n")


############################################################
# 3. 示例主函数：如何使用以上工具处理VOC分割、过分割、知识库匹配
############################################################
if __name__ == "__main__":
    # 路径示例（需自行替换为本地真实路径）
    voc_image_path_bike = "/data2/lihl/data/VOCdevkit/VOC2012/JPEGImages/2008_003270.jpg"
    voc_seg_path_bike   = "/data2/lihl/data/VOCdevkit/VOC2012/SegmentationClass/2008_003270.png"

    voc_image_path_horse = "/data2/lihl/data/VOCdevkit/VOC2012/JPEGImages/2008_000602.jpg"
    voc_seg_path_horse   = "/data2/lihl/data/VOCdevkit/VOC2012/SegmentationClass/2008_000602.png"
    save_root      = "/data/lihl/fsod/results"
    os.makedirs(save_root, exist_ok=True)

    # 假设我们想针对 "bicycle" 这个类做测试
    class_name = "bicycle"

    # 1) 初始化知识库
    kb = Knowledge_Base(
        device="cuda" if torch.cuda.is_available() else "cpu",
        clip_model_name="openai/clip-vit-base-patch32",
        similarity_threshold=0.3,
        segmentation_params={"n_segments": 100, "compactness": 0.1, "sigma": 3}
    )

    # 2) 准备文本属性
    #    例如: "bicycle: wheels, a frame, handlebars, a seat, and pedals."
    horse_attributes = "horse: ,a muscular build, a long mane, a flowing tail, slender legs, a large head, expressive eyes, pointed ears, and hard hooves."
    kb.process_attributes(horse_attributes)

    bird_attributes = "bird: ,a light skeleton, feathers covering the body, a beak, two wings, hollow bones, a streamlined body shape, and webbed feet."
    kb.process_attributes(bird_attributes)

    bicycle_attributes = "bicycle: ,wheel, a frame, handlebars, a seat, and pedals."
    kb.process_attributes(bicycle_attributes)

    # 3) 从VOC分割标注提取所有 "bicycle" 区域
    extracted_imgs = extract_voc_segments(
        voc_image_path_bike,
        voc_seg_path_bike,
        class_name=class_name,
        save_dir=os.path.join(save_root, "extracted_regions")  # 可选
    )
    print(f"Found {len(extracted_imgs)} connected regions for class '{class_name}'")

    # 4) 对每个提取区域执行 SLIC + 匹配，并将结果加入知识库
    for idx, region_img in enumerate(extracted_imgs):
        prefix_str = f"{class_name}_region{idx}"
        results = kb.process_cropped_image(
            region_img, 
            class_name=class_name, 
            save_dir=save_root,
            prefix=prefix_str
        )
        kb.add_to_knowledge(class_name, results)
        break
    
    class_name = 'horse'
    extracted_imgs = extract_voc_segments(
        voc_image_path_horse,
        voc_seg_path_horse,
        class_name='horse',
        save_dir=os.path.join(save_root, "extracted_regions")  # 可选
    )
    print(f"Found {len(extracted_imgs)} connected regions for class 'horse'")

    for idx, region_img in enumerate(extracted_imgs):
        prefix_str = f"{class_name}_region{idx}"
        results = kb.process_cropped_image(
            region_img, 
            class_name=class_name, 
            save_dir=save_root,
            prefix=prefix_str
        )
        kb.add_to_knowledge(class_name, results)

    # 查看知识库结构
    kb.print_structure()

    # 5) 保存知识库
    kb_path = os.path.join(save_root, "my_kb.pkl")
    kb.save(kb_path)
    print(f"Knowledge base saved to {kb_path}")

    # 6) 加载知识库并做查询
    kb2 = Knowledge_Base.load(kb_path, device="cpu")
    kb2.print_structure()

    # 假设想查询 "wheel" 的文本特征在 bicycle 里的相似patch
    query_text = "wheel"
    inputs = kb2.clip_processor(text=[query_text], return_tensors="pt", padding=True)
    for k in inputs:
        inputs[k] = inputs[k].to(kb2.device)
    with torch.no_grad():
        query_embed = kb2.clip_model.get_text_features(**inputs)[0]  # shape (D,)

    # 查询 top-k
    query_save_path = os.path.join(save_root, f"query_{query_text}.png")
    results = kb2.query(
        query_embed,
        class_name="bicycle",
        top_k=5,
        save_path=query_save_path  # 保存可视化查询结果
    )
    print("Query Results for 'wheel':")
    for (attr_name, sim_score, idx_in_attr_list) in results:
        print(f"  - {attr_name}, sim={sim_score:.4f}, idx={idx_in_attr_list}")
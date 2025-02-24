import cv2
import numpy as np
import os
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# VOC 数据集路径（需要更改为你的数据集路径）
VOC_ROOT = "/data2/lihl/data/VOCdevkit/VOC2007"
JPEGIMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")
SEGMENTATION_CLASS_DIR = os.path.join(VOC_ROOT, "SegmentationClass")

OUTPUT_DIR = "/data/lihl/fsod/model/knowledge_base/debug_sam"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. 读取分割掩码，获取目标区域
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

# 2. 进行 SLIC 超像素分割（仅针对 `horses`）
def segment_horse(image, mask, output_seg_path):
    # 只在 `horses` 区域进行分割
    horses_pixels = img_as_float(image[:, :, :3])  # 只取 RGB 通道
    segments = slic(horses_pixels, n_segments=10, compactness=5, sigma=1, mask=mask)

    # 可视化并保存
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(horses_pixels, segments))
    plt.axis('off')
    plt.savefig(output_seg_path, bbox_inches='tight')
    plt.close()
    return segments


# 3. 进行 CLIP 计算匹配，并可视化小图
def clip_match_and_visualize(image, segments, class_labels, output_match_path):
    matched_results = []

    for label in np.unique(segments):
        mask = (segments == label).astype(np.uint8) * 255  # 当前超像素区域的掩码

        # 计算区域边界
        y, x = np.where(mask)
        if len(y) < 10 or len(x) < 10:
            continue

        # **像素级裁剪**，背景透明
        patch = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        patch[:, :, :3] = image[:, :, :3]
        patch[:, :, 3] = mask  # 透明背景
        
        # 只保留非零区域
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        patch = patch[ymin:ymax, xmin:xmax, :]

        # CLIP 处理（仅输入 **透明背景的分割 patch**）
        pil_patch = Image.fromarray(patch).convert("RGBA")
        inputs = processor(images=pil_patch, return_tensors="pt", padding=True).to(device)
        image_features = model.get_image_features(**inputs)

        text_inputs = processor(text=class_labels, images=pil_patch, return_tensors="pt", padding=True).to(device)
        outputs = model(**text_inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        best_match_idx = probs.argmax().item()
        best_match_label = class_labels[best_match_idx]
        best_match_score = probs[0][best_match_idx].item()

        # 存储匹配结果
        if label == 0:
            matched_results.append((patch, 'Background', 0))
        else:
            matched_results.append((patch, best_match_label, best_match_score))

    # 创建网格可视化
    num_results = len(matched_results)
    cols = 5
    rows = (num_results // cols) + (1 if num_results % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, (patch, label, score) in enumerate(matched_results):
        axes[i].imshow(patch)
        axes[i].set_title(f"{label}\n{score:.4f}", fontsize=10)
        axes[i].axis("off")

    # 移除多余的空白子图
    for i in range(num_results, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_match_path, bbox_inches="tight")
    plt.close()

# 3. 进行 CLIP 计算匹配，并可视化小图
def clip_match_and_visualize_v2(image, segments, class_labels, output_match_path):
    matched_results = []

    for label in np.unique(segments):
        mask = (segments == label).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        y, x = np.where(mask)
        if len(y) < 10 or len(x) < 10:
            continue
        region = image[np.min(y):np.max(y), np.min(x):np.max(x), :3]

        # CLIP 处理
        pil_image = Image.fromarray(region).convert("RGB")
        inputs = processor(images=pil_image, return_tensors="pt", padding=True).to(device)
        image_features = model.get_image_features(**inputs)

        text_inputs = processor(text=class_labels, images=pil_image, return_tensors="pt", padding=True).to(device)
        outputs = model(**text_inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        best_match_idx = probs.argmax().item()
        best_match_label = class_labels[best_match_idx]
        best_match_score = probs[0][best_match_idx].item()

        # 存储匹配结果
        matched_results.append((region, best_match_label, best_match_score))

    # 创建多子图可视化
    num_results = len(matched_results)
    cols = 5
    rows = (num_results // cols) + (1 if num_results % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, (region, label, score) in enumerate(matched_results):
        axes[i].imshow(region)
        axes[i].set_title(f"{label}\n{score:.4f}", fontsize=10)
        axes[i].axis("off")

    # 移除多余的空白子图
    for i in range(num_results, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_match_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # 选择一张图片
    image_id = "001420"
    input_image = os.path.join(JPEGIMAGES_DIR, f"{image_id}.jpg")
    mask_image = os.path.join(SEGMENTATION_CLASS_DIR, f"{image_id}.png")

    output_cropped_image = os.path.join(OUTPUT_DIR, f"{image_id}_horses.png")  # 透明背景
    output_cropped_mask = os.path.join(OUTPUT_DIR, f"{image_id}_horses_mask.png")
    output_seg = os.path.join(OUTPUT_DIR, f"{image_id}_segmentation_result.png")
    output_match = os.path.join(OUTPUT_DIR, f"{image_id}_match_result.jpg")
    output_match_v2 = os.path.join(OUTPUT_DIR, f"{image_id}_match_result_v2.jpg")
    input_text = "Horses: a muscular build, a long mane, a flowing tail, slender legs, a large head, expressive eyes, pointed ears, hard hooves"
    horses_related_labels = ["horse's muscular body", "long mane of horse", "flowing tail of horse", "slender legs of horse", "large head of horse",
                             "expressive eyes of horse", "pointed ears of horse", "hard hooves of horse"]
    # 1. 提取 `horses` 类像素级区域
    cropped_img, cropped_mask = extract_horses_pixels(input_image, mask_image)

    if cropped_img is None:
        print(f"未找到 {image_id} 的 horses 类目标，跳过处理")
    else:
        # 保存裁剪后的 `horses`
        cv2.imwrite(output_cropped_image, cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2BGRA))  # 透明背景
        cv2.imwrite(output_cropped_mask, cropped_mask)

        # 2. 仅对 `horses` 进行超像素分割
        segments = segment_horse(cropped_img, cropped_mask, output_seg)

        # 3. CLIP 计算匹配
        clip_match_and_visualize(cropped_img, segments, horses_related_labels, output_match)
        clip_match_and_visualize_v2(cropped_img, segments, horses_related_labels, output_match_v2)
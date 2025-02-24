import json, re, os
import numpy as np
import torch, json, argparse
import pandas as pd
from tqdm import tqdm, trange
from data.get_sft_dataset import Get_SFT_Dataset
from data.util import  load_data_from_batches
from model.image_encoder import feat_extractor
from model.util import load_lora_model, load_ft_model

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='llama3.1-8b', help='name of llm model like llama3.1-8b')
parser.add_argument('--finetune-mode', type=str, default='lora', help='test mode: lora or ft_full')
parser.add_argument('--data-path', type=str, help='training batch folder')
parser.add_argument('--processed-voc-data', type=str, default='/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset.jsonl', help="jsonl format file, include image_path, image_size, annotation, class")
parser.add_argument('--prepare-description', type=bool, default=False, help='need describe the image? True or False, default false.')
parser.add_argument('--max-seq-length', type=int, help='the max sequence length after tokenize the inputs')
parser.add_argument('--lora-path', type=str, help='the lora path')
args = parser.parse_args()

def read_json(json_string):
    fixed = re.sub(r',\s*([\]}])', r'\1', json_string)
    try:
        parsed_data = json.loads(fixed)
        return parsed_data
    except json.JSONDecodeError:
        print("Warning: Cannot parse the string as JSON, returning None instead.")
        print(fixed)
        return None

def compute_iou(bbox1, bbox2):
    """
    计算两个边界框的 IoU
    bbox = [xmin, ymin, xmax, ymax]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def voc_ap(recalls, precisions):
    """
    使用类似 VOC 的插值方式，计算 AP:
     - 将 precision 修正为单调递减
     - 对 recall 轴进行积分
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # 让 precision 变为单调递减
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # 对 recall 的每个区间，乘以区间对应的 precision
    ap = 0.0
    for i in range(len(mrec) - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    
    return ap

def compute_ap_for_class(gt_list, pred_list, iou_thr=0.5):
    """
    计算单个类别在指定IoU阈值下的 AP。
    - gt_list: [(image_id, [xmin,ymin,xmax,ymax]), ...]
    - pred_list: [(image_id, [xmin,ymin,xmax,ymax], score), ...]
    - iou_thr: IoU匹配阈值
    """
    if len(gt_list) == 0:
        # 没有GT，如果也没有pred，AP=0; 有pred则也只能算0
        return 0.0
    
    # 按照score从高到低排序
    pred_list = sorted(pred_list, key=lambda x: x[2], reverse=True)
    
    # 每个GT只能被匹配一次，记录已匹配情况: dict: (image_id, gt_idx) -> bool
    # 先把 gt_list 按 image_id 分组
    gt_map = {}
    for idx, (img_id, gt_bbox) in enumerate(gt_list):
        gt_map.setdefault(img_id, []).append({"bbox": gt_bbox, "matched": False})
    
    tp_list = []
    fp_list = []
    
    for (img_id, pred_bbox, score) in pred_list:
        if img_id not in gt_map:
            # 该图像没有任何GT
            tp_list.append(0)
            fp_list.append(1)
            continue
        
        gt_candidates = gt_map[img_id]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # 找到 IoU 最大的未匹配 GT
        for g_i, gt_item in enumerate(gt_candidates):
            if not gt_item["matched"]:
                iou = compute_iou(pred_bbox, gt_item["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_i
        
        if best_iou >= iou_thr and best_gt_idx != -1:
            # 匹配成功
            tp_list.append(1)
            fp_list.append(0)
            gt_candidates[best_gt_idx]["matched"] = True
        else:
            # 匹配失败
            tp_list.append(0)
            fp_list.append(1)
    
    # 计算 Precision / Recall
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    total_gt = len(gt_list)  # 该类别的 GT 数量
    
    precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
    recalls = tp_cum / (total_gt + 1e-16)
    
    ap = voc_ap(recalls, precisions)
    return ap

def compute_map_multiple_images(gt_data, pred_data, iou_thresholds=None):
    """
    针对多张图，计算 mAP（包含 0.5 到 0.95 的多阈值）。
    gt_data: List[ {image_id: str, annotations: [...]} ]
    pred_data: List[ {image_id: str, annotations: [...]} ]
    
    每张图的 annotations 里：
      - GT: { "class": str, "bbox": [xmin,ymin,xmax,ymax] }
      - Pred: { "class": str, "bbox": [...], "score": float }
    
    返回: {
      "mAP_0.5": xx,
      "mAP_0.5:0.95": xx,
      "AP_per_iou": {
         0.5: xx,
         0.55: xx,
         ...
      }
    }
    """
    if iou_thresholds is None:
        iou_thresholds = [round(iou, 2) for iou in np.arange(0.5, 1.0, 0.05)]
    
    # 1) 按类别收集所有GT和Pred
    #    gt_dict[class_name] = [(image_id, bbox), ...]
    #    pred_dict[class_name] = [(image_id, bbox, score), ...]
    gt_dict = {}
    pred_dict = {}
    
    # 读取多张图的 GT
    for item in gt_data:
        img_id = item["image_id"]
        annotations = item["annotations"]
        for ann in annotations:
            cls = ann["class"]
            bbox = ann["bbox"]  # [xmin,ymin,xmax,ymax]
            gt_dict.setdefault(cls, []).append((img_id, bbox))
    
    # 读取多张图的预测
    for item in pred_data:
        img_id = item["image_id"]
        annotations = item["annotations"]
        for ann in annotations:
            cls = ann["class"]
            bbox = ann["bbox"]
            score = ann.get("score", 1.0)  # 如果没有score，就默认1.0
            pred_dict.setdefault(cls, []).append((img_id, bbox, score))
    
    # 所有类别的合集
    all_classes = set(list(gt_dict.keys()) + list(pred_dict.keys()))
    
    # 2) 逐个 IoU 阈值计算 AP，最后求平均
    ap_per_iou = {}
    for iou_thr in iou_thresholds:
        ap_list = []
        for cls in all_classes:
            gt_list = gt_dict.get(cls, [])
            pred_list = pred_dict.get(cls, [])
            ap = compute_ap_for_class(gt_list, pred_list, iou_thr=iou_thr)
            ap_list.append(ap)
        
        #if len(ap_list) > 0:
        #    ap_per_iou[iou_thr] = np.mean(ap_list)
        #else:
        #    ap_per_iou[iou_thr] = 0.0
        rounded_iou = round(iou_thr, 2)  # 键保留两位小数
        mean_ap = round(np.mean(ap_list), 4) if ap_list else 0.0  # 值保留四位小数
        ap_per_iou[rounded_iou] = mean_ap
    
    mAP_0_5 = ap_per_iou.get(0.5, 0.0)  # 直接使用四舍五入后的键 0.5
    mAP_0_5_to_0_95 = round(np.mean(list(ap_per_iou.values())), 4)  # 最终结果也保留四位小数
    
    return {
        "mAP_0.5": mAP_0_5,
        "mAP_0.5:0.95": mAP_0_5_to_0_95,
        "AP_per_iou": ap_per_iou
    }

def extract_annotations(text):
    # 分割文本并保留原始结构
    raw_parts = text.split('<|eot_id|>')
    
    # 逆向搜索包含assistant标记的有效块
    target_idx = None
    for i in reversed(range(len(raw_parts))):
        part = raw_parts[i].strip()
        if '<|start_header_id|>assistant<|end_header_id|>' in part:
            target_idx = i
            break
    
    if target_idx is None:
        return text, {}  # 未找到目标内容

    # 提取目标块并分离头部
    target_part = raw_parts[target_idx].strip()
    header, _, json_content = target_part.partition('<|end_header_id|>')
    json_content = json_content.strip()

    # JSON修复逻辑
    fixes = [
        (r',\s*([\]}])', r'\1'),          # 去除末尾逗号
        (r'/\*.*?\*/', '', re.DOTALL),    # 删除块注释
        (r'//.*', '')                    # 删除行注释
        #(r'"(?!\b(class|bbox)\b)[^"]*":', '"unknown":')  # 处理非法键名
    ]
    
    for pattern, repl, *flags in fixes:
        json_content = re.sub(pattern, repl, json_content, flags=flags[0] if flags else 0)

    try:
        extracted = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败：{str(e)}")
        return text, {}

    remaining_parts = raw_parts[:target_idx]
    remaining_parts.append(header + '<|end_header_id|>')  # 保留处理后的头部
    remaining_text = '<|eot_id|>'.join(remaining_parts)
    
    if not remaining_text.endswith('<|eot_id|>'):
        remaining_text += '<|eot_id|>'

    return remaining_text.strip(), extracted
    
def model_test(args):
    if args.finetune_mode == 'ft_full':
        model, tokenizer = load_ft_model()
    elif args.finetune_mode == 'lora':
        model, tokenizer = load_lora_model(model_path=args.lora_path)
    else:
        raise ValueError("please choose a test model")
    print(f'testing {args.finetune_mode} model in {args.lora_path}')
    predictions = [] # json的list
    ground_truths = []  # json的list
    subfolders = ['train', 'test']
    for subfolder in subfolders:
        folder_path = os.path.join(args.data_path, subfolder)  # 构建完整路径
        data_list =  load_data_from_batches(folder_path)
        print(f'testing {folder_path}')
        for idx in range(len(data_list)):
            decoded_text = tokenizer.decode(data_list[idx]['input_ids'][0])
            #prompt, label = get_input_label(data_list[idx]['input_ids'][0], tokenizer)
            #print(decoded_text)
            prompt, label = extract_annotations(decoded_text)
            
            tokenized = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=False,
                        truncation=True
                    ).to("cuda")

            after_input_ids = tokenized["input_ids"]
            after_input_ids = after_input_ids[:, 1:]
            sup_visual_tokens = data_list[idx]['sup_visual_tokens']
            que_visual_tokens = data_list[idx]['que_visual_tokens']
            eos_token_id = 128009
            with torch.no_grad():
                outputs = model.generate(
                    #input_ids=inputs["input_ids"],
                    #attention_mask=inputs["attention_mask"],
                    inputs=after_input_ids,
                    sup_visual_tokens=sup_visual_tokens,
                    que_visual_tokens=que_visual_tokens,
                    #max_length=512,
                    max_new_tokens=256,
                    do_sample=True,
                    eos_token_id=eos_token_id,
                    top_p=0.9,
                    temperature=0.3
                )
                #print(outputs.size())
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                #print(f'Generate text[0]: {generated_text[0]}')
                pred = read_json(generated_text[0])
                gt = label
                #print(f'pred:{pred}')
                #print(f'gt:{gt}')
            if pred is not None:
                pred["image_id"] = f"img{idx}.jpg"
                gt["image_id"] = f"img{idx}.jpg"
                    
                ground_truths.append(gt)
                predictions.append(pred)
            if idx % 2000 == 0:
                print(f"testing: {idx}/{len(data_list)}")
            if idx == 200:
                break
        print(f'--[{subfolder}] {args.lora_path} result:')
        results = compute_map_multiple_images(ground_truths, predictions)
        print("  mAP@0.5:      ", results["mAP_0.5"])
        print("  mAP@0.5:0.95: ", results["mAP_0.5:0.95"])
        print("  AP_per_iou:   ", results["AP_per_iou"])
    return


if __name__ == "__main__":
    model_test(args)
    # 多张图的 GT 列表
    # 每个元素表示一张图片的标注，包含 image_id 和 annotations
    gt_data = [
        {
            "image_id": "img1.jpg",
            "annotations": [
                {"class": "car", "bbox": [220, 256, 410, 348]},
                {"class": "car", "bbox": [21, 288, 134, 375]},
                {"class": "car", "bbox": [9, 269, 42, 287]}
            ]
        },
        {
            "image_id": "img2.jpg",
            "annotations": [
                {"class": "car", "bbox": [100, 100, 200, 200]},
                {"class": "person", "bbox": [50, 50, 90, 120]}
            ]
        }
    ]

    # 多张图的预测列表
    pred_data = [
        {
            "image_id": "img1.jpg",
            "annotations": [
                {"class": "car", "bbox": [219, 234, 400, 350], "score": 0.9},
                {"class": "car", "bbox": [21, 288, 134, 375], "score": 0.8},
                {"class": "car", "bbox": [9, 269, 43, 287],  "score": 0.7}
            ]
        },
        {
            "image_id": "img2.jpg",
            "annotations": [
                {"class": "car",    "bbox": [95, 105, 205, 195], "score": 0.95},
                {"class": "person", "bbox": [48, 48, 89, 121],   "score": 0.6}
            ]
        }
    ]
    
    #results = compute_map_multiple_images(gt_data, pred_data)
    #print("mAP@0.5:      ", results["mAP_0.5"])
    #print("mAP@0.5:0.95: ", results["mAP_0.5:0.95"])
    #print("AP_per_iou:   ", results["AP_per_iou"])

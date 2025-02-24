import torch, json, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.metrics import accuracy_score
from data.get_sft_dataset import Get_SFT_Dataset
from data.util import  load_data_from_batches
from model.image_encoder import feat_extractor
from model.util import load_lora_model, load_ft_model
from pycocotools import coco
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='llama3.1-8b', help='name of llm model like llama3.1-8b')
parser.add_argument('--finetune-mode', type=str, default='lora', help='test mode: lora or ft_full')
parser.add_argument('--data-path', type=str, help='training batch folder')
parser.add_argument('--processed-voc-data', type=str, default='/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset.jsonl', help="jsonl format file, include image_path, image_size, annotation, class")
parser.add_argument('--prepare-description', type=bool, default=False, help='need describe the image? True or False, default false.')
parser.add_argument('--max-seq-length', type=int, help='the max sequence length after tokenize the inputs')
parser.add_argument('--lora-path', type=str, help='the lora path')
args = parser.parse_args()

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_ap(predictions, ground_truths, iou_threshold):
    """计算某个IOU阈值下的AP值"""
    # 按置信度排序预测结果
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    # 初始化变量
    num_gts = len(ground_truths)
    detected_gts = [False] * num_gts
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # 对每个预测框计算TP或FP
    for pred_idx, pred in enumerate(predictions):
        pred_bbox = pred[0]
        pred_img_id = pred[2]
        
        max_iou = 0
        max_idx = -1
        
        # 找到同一图片中IOU最大的ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if gt[1] != pred_img_id:
                continue
            
            iou = calculate_iou(pred_bbox, gt[0])
            if iou > max_iou:
                max_iou = iou
                max_idx = gt_idx
        
        # 根据IOU阈值判定是否为TP
        if max_iou >= iou_threshold and not detected_gts[max_idx]:
            tp[pred_idx] = 1
            detected_gts[max_idx] = True
        else:
            fp[pred_idx] = 1
    
    # 计算累积TP和FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # 计算precision和recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gts
    
    # 添加起始点
    precision = np.concatenate(([1], precision))
    recall = np.concatenate(([0], recall))
    
    # 计算AP (使用11点插值法)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
        
    return ap

def calculate_classification_accuracy(predictions, ground_truths):
    """计算分类准确率"""
    y_true = []
    y_pred = []

    for class_id in predictions.keys():
        pred = predictions[class_id] 
        gt = ground_truths[class_id] 

        for (pred_bbox, pred_confidence), (gt_bbox, gt_class, _) in zip(pred, gt):

            pred_class = class_id  
            gt_class_true = gt_class  

            y_true.append(gt_class_true)
            y_pred.append(pred_class)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def read_json(json_string):
    try:
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError:
        print("Warning: Cannot parse the string as JSON, returning None instead.")
        print(json_string)
        return None

def model_test(args):
    df = pd.read_json(args.processed_voc_data, lines=True)
    filtered_df = df[df['class'] == 'bicycle']
    filtered_df = filtered_df.reset_index(drop=True)
    feat_ext = feat_extractor()
    if args.finetune_mode == 'ft_full':
        model, tokenizer = load_ft_model()
    elif args.finetune_mode == 'lora':
        model, tokenizer = load_lora_model(model_path=args.lora_path)
    else:
        raise ValueError("please choose a test model")
    print(f'testing {args.finetune_mode} model')
    DATASET = Get_SFT_Dataset(model_name=args.model_name, dataset=filtered_df.to_dict(orient='records'), args=args)
    
    predictions = []  # [(bbox, confidence, image_id), ...]
    ground_truths = []  # [(bbox, image_id), ...]
    correct_classifications = 0
    total_samples = len(filtered_df)
    
    for idx, row in tqdm(filtered_df.iterrows(), total=(total_samples)):
        cls_gt = row['class']
        img_path = row['image_path']
        bbox_gt = row['annotation']
        img_size = row['image_size']
        input_ids, sup_visual_tokens, que_visual_tokens = DATASET.prepare_test_data(
            cls_name=cls_gt, que_image_path=img_path, que_image_size=img_size, idx=idx
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                sup_visual_tokens=sup_visual_tokens,
                que_visual_tokens=que_visual_tokens,
                max_length=512,
                do_sample=True,
                temperature=0.7,
            )
        
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        data_infered = read_json(generated_text[0])

        if data_infered == None:
            continue

        pred_class = data_infered['class']
        pred_bbox = data_infered['bounding_box']

        # 计算分类正确的样本数
        if pred_class == cls_gt:
            correct_classifications += 1
        
        # 添加预测和真实标签到列表中
        predictions.append((pred_bbox, 1.0, idx))  # 假设置信度为1.0
        ground_truths.append((bbox_gt, idx))

    # 计算MAP@50和MAP@95
    map50 = calculate_ap(predictions, ground_truths, iou_threshold=0.5)
    map95 = calculate_ap(predictions, ground_truths, iou_threshold=0.95)

    # 计算分类准确率
    classification_accuracy = correct_classifications / total_samples
    
    '''
    # 计算MAP@95 (平均多个IOU阈值的AP)
    aps = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        ap = calculate_ap(predictions, ground_truths, iou_threshold=iou_thresh)
        aps.append(ap)
    map95 = np.mean(aps)
    '''
    
    print(f"test mAP@0.5: {map50}")
    print(f"test mAP@0.95: {map95}")
    print(f"test Classification Accuracy: {classification_accuracy}")

    return

def model_test_v2(args):
    data_list =  load_data_from_batches(args.data_path)
    model, tokenizer = load_ft_model()

    predictions = []  # [(bbox, confidence, image_id), ...]
    ground_truths = []  # [(bbox, image_id), ...]
    correct_classifications = 0
    total_samples = len(data_list)

    for idx in trange(len(data_list)):
        input_ids = data_list[idx]['input_ids']
        attention_mask = data_list[idx]['attention_mask']
        sup_visual_tokens = data_list[idx]['sup_visual_tokens']
        que_visual_tokens = data_list[idx]['que_visual_tokens']
        labels = data_list[idx]['labels']
        filtered_labels = labels.clone()
        filtered_labels[labels == -100] = tokenizer.pad_token_id  # 替换为 pad_token_id
        decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                sup_visual_tokens=sup_visual_tokens,
                que_visual_tokens=que_visual_tokens,
                max_length=512,
                do_sample=True,
                temperature=0.7,
            )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        data_infered = read_json(generated_text[0])
        data_gt = read_json(decoded_labels[0])

        cls_gt = data_gt['class']
        bbox_gt = data_gt['bounding_box']

        pred_class = data_infered['class']
        pred_bbox = data_infered['bounding_box']

        # 计算分类正确的样本数
        if pred_class == cls_gt:
            correct_classifications += 1
        
        # 添加预测和真实标签到列表中
        predictions.append((pred_bbox, 1.0, idx))  # 假设置信度为1.0
        ground_truths.append((bbox_gt, idx))

    # 计算MAP@50和MAP@95
    map50 = calculate_ap(predictions, ground_truths, iou_threshold=0.5)
    map95 = calculate_ap(predictions, ground_truths, iou_threshold=0.95)

    # 计算分类准确率
    classification_accuracy = correct_classifications / total_samples
    
    '''
    # 计算MAP@95 (平均多个IOU阈值的AP)
    aps = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        ap = calculate_ap(predictions, ground_truths, iou_threshold=iou_thresh)
        aps.append(ap)
    map95 = np.mean(aps)
    '''
    
    print(f"test mAP@0.5: {map50}")
    print(f"test mAP@0.95: {map95}")
    print(f"test Classification Accuracy: {classification_accuracy}")




if __name__ == '__main__':
    
    #model_test(args)
    #model_test_v2(args)

    for i in range(3):
        print('-'*50, f'test id: {i}', '-'*50)
        model_test(args)

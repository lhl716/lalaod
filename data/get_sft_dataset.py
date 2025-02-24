import os, random, torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
import xml.etree.ElementTree as ET
from model.LLaVA.run_llava_class import LLAVA
from model.LLaVA.llava.mm_utils import get_model_name_from_path
from model.image_encoder import feat_extractor
from data.util import save_data_in_batches
from data.build_data.config import get_train_data, get_test_data, get_train_data_for_llama2
from data.build_data.voc import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES

def parse_voc_annotation(annotation_path):
    """
    解析 VOC 数据集的 XML 标注文件，提取类别、边界框信息和图像尺寸。
    
    Args:
        annotation_path (str): VOC 数据集中的 XML 标注文件路径。
    
    Returns:
        dict: 包含图像尺寸、类别、边界框信息的字典。
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'class': cls_name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return {
        'image_size': (width, height),
        'objects': objects
    }

def choose_sup_img(dataset, cls_name, fixed_index=None):
        """
        从给定的数据集中，按照指定的类别随机或固定选择一个 file_path。

        Args:
            dataset (list): 数据集，每条数据是一个字典，包含 'image_path', 'class', 'annotation' 等信息。
            cls_name (str): 要选择的类别名称。
            fixed_index (int, optional): 如果提供，则按固定索引选择；否则随机选择。

        Returns:
            str: 选定的 file_path。如果没有匹配的文件，返回 None。
        """
        filtered = [entry for entry in dataset if entry['class'] == cls_name]
        if not filtered:
            print(f"No entries found for class: {cls_name}")
            return None

        if fixed_index is not None:
            index = fixed_index % len(filtered)  # 防止索引越界
        else:
            index = random.randint(0, len(filtered) - 1)  # 随机选择

        print(f'"{cls_name}":"{index}",')

def prepare_voc_dataset_without_attr(file_path, split_id, split="trainval"):
    """
    准备 VOC 数据集的训练/测试数据。

    Args:
        file_path (str): VOC 数据集根目录路径。
        year (str): VOC 数据集年份，例如 "2007"。
        split (str): 数据集划分，例如 "trainval" 或 "test"。

    Returns:
        list: 包含每条数据的信息，格式为：
              [{'image_path': ..., 'class': ..., 'annotation': ...}, ...]
    """
    base_classes = PASCAL_VOC_BASE_CATEGORIES[split_id]
    novel_classes = PASCAL_VOC_NOVEL_CATEGORIES[split_id]
    all_classes = PASCAL_VOC_ALL_CATEGORIES[split_id]
    base_class_dataset = []
    novel_class_dataset = []
    all_class_dataset = []
    
    years=["2007","2012"]

    non_image_list =[]
    non_annotation_list = []
    for year in years:
        image_set_path = os.path.join(file_path, f"VOC{year}/ImageSets/Main/{split}.txt")
        image_dir = os.path.join(file_path, f"VOC{year}/JPEGImages")
        annotation_dir = os.path.join(file_path, f"VOC{year}/Annotations")

        # 读取图像 ID 列表
        with open(image_set_path, 'r') as f:
            image_ids = f.read().splitlines()

        for image_id in image_ids:
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            annotation_path = os.path.join(annotation_dir, f"{image_id}.xml")

            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                #print(f"Warning: Missing file for {image_path}. Skipping.")
                non_image_list.append(image_path)
                non_annotation_list.append(annotation_path)
                continue
            
            info = parse_voc_annotation(annotation_path)
            objects = info['objects']

            for obj in objects:
                cls_name = obj['class']
                data_entry = {
                    'image_path': image_path,
                    'class': cls_name,
                    'annotation': obj['bbox'],
                    'image_size': info['image_size']
                }
                if not os.path.exists(image_path):
                    print(cls_name)
                if cls_name in base_classes:
                    base_class_dataset.append(data_entry)
                elif cls_name in novel_classes:
                    novel_class_dataset.append(data_entry)
                all_class_dataset.append(data_entry)
    df_base = pd.DataFrame(base_class_dataset)
    df_novel = pd.DataFrame(novel_class_dataset)
    df_all = pd.DataFrame(all_class_dataset)
    
    df_base.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/base_class_dataset.jsonl', orient='records', lines=True)
    df_novel.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/novel_class_dataset.jsonl', orient='records', lines=True)
    df_all.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset.jsonl', orient='records', lines=True)
    print(f'未找到图片{len(non_image_list)}')
    return all_class_dataset, base_class_dataset, novel_class_dataset

class Get_SFT_Dataset():
    def __init__(self, model_name, dataset, args=None, llava_args=None):
        self.feat_ext = feat_extractor()
        self.dataset = dataset
        self.args = args
        print(f'Dataset length: {len(self.dataset)}')
        if model_name == 'llama3.1-8b':
            self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif model_name == 'llama2-7b':
            self.model_name  = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token


        if "<visual_sup>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_sup>', '<visual_que>']})
            #self.model.resize_token_embeddings(len(self.tokenizer))  # 调整模型的词汇表大小
        
        if args.prepare_description == True:
            self.llava_model_path = 'liuhaotian/llava-v1.5-13b'
            self.llava_model_name = get_model_name_from_path(self.llava_model_path)
            if llava_args:
                self.llava_args = llava_args
            else:
                self.llava_args = type('Args', (), {
                    "model_path": self.llava_model_path,
                    "model_base": None,
                    "model_name": self.llava_model_name,
                    "conv_mode": None,
                    "sep": ",",
                    "temperature": 0.7,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512,
                    "do_sample": True
                })()
            print("Loading llava...")
            self.llava = LLAVA(args)
        else:
            self.llava = None

    def choose_sup_img(self, cls_name, fixed_index=None):
        filtered = [entry for entry in self.dataset if entry['class'] == cls_name]
        if not filtered:
            print(f"No entries found for class: {cls_name}")
            return None

        if fixed_index is not None:
            index = fixed_index % len(filtered)  # 防止索引越界
        else:
            index = random.randint(0, len(filtered) - 1)  # 随机选择

        #print(f'cls_name:{cls_name}, index:{index}')
        return filtered[index]['image_path'], filtered[index]['annotation'], filtered[index]['image_size']
    
    def get_description(self, image_file, prompt, idx):
        if self.llava:
            outputs = self.llava.eval_model(prompt, image_file)
        else:
            outputs = self.dataset[idx]['description']
        return outputs
    
    def get_visual_tokens(self, image_path, idx):
        image = Image.open(image_path).convert("RGB")
        prompt =  "Describe this image in one sentence."
        caption = self.get_description(image_file=image_path, prompt=prompt, idx=idx)
        visual_tokens = self.feat_ext.forward(raw_img=image, caption=caption)
        return visual_tokens['multimodal_embeds'].detach().cpu().numpy()
        
    def prepare_dataset(self):
        #df = pd.read_json(self.dataset_path, lines=True)

        df = pd.DataFrame(self.dataset)
        df = df[df['class'] == 'bicycle']
        df = df.reset_index(drop=True)
        dataset=[]
        data_dict_count = 0
        if self.args.max_seq_length is not None:
                max_seq_length = self.args.max_seq_length
        else:
            max_seq_length = 400  # 固定的最大序列长度
        print(f"Padding data, max_seq_length: {max_seq_length}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing VOC Data"):

            #if row['class'] != 'bicycle':
            #    continue
            cls_name, que_image_path, que_annotation, que_image_size = row['class'], row['image_path'], row['annotation'], row['image_size']
            sup_image_path, sup_annotation, sup_image_size = self.choose_sup_img(cls_name, fixed_index=0)
            
            if sup_image_path == que_image_path: # 相同就再选一次
                sup_image_path, sup_annotation, sup_image_size = self.choose_sup_img(cls_name)
            
            sup_visual_tokens = torch.tensor(self.get_visual_tokens(sup_image_path, idx)).to("cuda")
            que_visual_tokens = torch.tensor(self.get_visual_tokens(que_image_path, idx)).to("cuda")
            
            if self.model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
                instruction, input, output = get_train_data(
                    sup_image_size, sup_annotation, que_image_size, que_annotation, cls_name
                )
            elif self.model_name == 'meta-llama/Llama-2-7b-chat-hf':
                instruction, input, output = get_train_data_for_llama2(
                    sup_image_size, sup_annotation, que_image_size, que_annotation, cls_name
                )

            prompt = instruction + input
            combined_text = prompt + output  

            #tokenized = self.tokenizer(
            #    combined_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length
            #).to("cuda")

            tokenized = self.tokenizer(
                combined_text,
                return_tensors="pt",  # 返回 PyTorch 张量
                padding=False,         # 禁用填充
                truncation=True,      # 启用截断，防止超出最大长度
                max_length=max_seq_length  # 最大长度
            ).to("cuda")

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.clone()

            prompt_length = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            #instruction_length = len(self.tokenizer(instruction, return_tensors="pt")["input_ids"][0])

            labels[:, :prompt_length] = -100 
            labels[labels == self.tokenizer.pad_token_id] = -100 

            #output_length = len(self.tokenizer(output, return_tensors="pt")["input_ids"][0])
            # 确定 output 部分的结束位置
            #output_end_idx = len(self.tokenizer(combined_text, return_tensors="pt")["input_ids"][0])

            #seq_length = input_ids.size(1)
            #padding_length = max_seq_length - seq_length

            assert input_ids.shape == labels.shape, "Input IDs and Labels must have the same shape"
            dataset.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask, # 做attention的函数，传递attention mask打印。
                "labels": labels,
                "sup_visual_tokens": sup_visual_tokens,
                "que_visual_tokens": que_visual_tokens
            })

            if len(dataset) == 10000:
                save_data_in_batches(dataset, output_dir=self.args.data_path, batch_size=len(dataset), start_batch_idx=data_dict_count)
                dataset = []
                data_dict_count += 1

        dataset_save_path = self.args.data_path
        #self.dataset = dataset
        save_data_in_batches(dataset, output_dir=dataset_save_path, batch_size=len(dataset), start_batch_idx=data_dict_count)
        print(f'Dataset saved to {dataset_save_path}')
    
    def prepare_test_data(self, cls_name, que_image_path, que_image_size, idx):
        
        #max_seq_length = 400  # 固定的最大序列长度
        if self.args.max_seq_length is not None:
            max_seq_length = self.args.max_seq_length
        else:
            max_seq_length = 400
        sup_image_path, sup_annotation, sup_image_size = self.choose_sup_img(cls_name, fixed_index=0)
        
        if sup_image_path == que_image_path: # 相同就再选一次
            sup_image_path, sup_annotation, sup_image_size = self.choose_sup_img(cls_name)
        
        #sup_visual_tokens = torch.tensor(self.get_visual_tokens(sup_image_path, idx)).to("cuda")
        sup_visual_tokens = None
        que_visual_tokens = torch.tensor(self.get_visual_tokens(que_image_path, idx)).to("cuda")
        
        instruction, input = get_test_data(
            sup_image_size, sup_annotation, que_image_size, cls_name
        )

        prompt = instruction + input

        tokenized = self.tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length
        ).to("cuda")

        input_ids = tokenized["input_ids"]

        return input_ids, sup_visual_tokens, que_visual_tokens
    
    def prepare_json_described_data(self):
        df = pd.read_json(self.args.processed_voc_dats, lines=True)
        descriptions=[]
        print("Describing image...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            prompt = f"This is a picture with {row['class']} in it, describe in one sentence what is in this picture."
            description = self.get_description(image_file=row['image_path'], prompt=prompt)
            descriptions.append(description)
        
        df['description'] = descriptions
        df.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset_with_description.jsonl', orient='records', lines=True)


import os, random, torch, json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
import xml.etree.ElementTree as ET
from model.LLaVA.run_llava_class import LLAVA
from model.LLaVA.llava.mm_utils import get_model_name_from_path
from model.image_encoder import feat_extractor
from data.util import save_data_in_batches
from data.build_data.config import get_train_data, get_test_data, get_train_data_for_llama2, get_prompt_new
from data.build_data.voc import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES

def build_sup_data_map(dataset):
    """
    构建映射:  cls_name -> [所有“包含过这个类”的 sup_data]
    注意，这里不要求 sup_data 只有该类；只要它包含过 cls_name 即可。
    """
    sup_data_map = {}
    for data in dataset:
        distinct_classes = {ann["category_name"] for ann in data["annotations"]}
        for cls in distinct_classes:
            if cls not in sup_data_map:
                sup_data_map[cls] = []
            sup_data_map[cls].append(data)
    return sup_data_map

def filter_data_for_class(data, cls_name):
    """
    只保留 data 里属于 cls_name 的 annotation。
    用于在给 get_prompt_new() 时，只传递和 cls_name 相关的 bbox。
    """
    filtered_annotations = [
        ann for ann in data["annotations"] 
        if ann["category_name"] == cls_name
    ]
    new_data = {
        "image_path":   data["image_path"],
        "image_size":   data["image_size"],
        "description":  data["description"],
        "annotations":  filtered_annotations
    }
    return new_data

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
    
    # df_base.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/base_class_dataset.jsonl', orient='records', lines=True)
    # df_novel.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/novel_class_dataset.jsonl', orient='records', lines=True)
    # df_all.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset.jsonl', orient='records', lines=True)
    print(f'未找到图片{len(non_image_list)}')
    regular_dataset(data_path='/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset_with_description.jsonl', output_path='/data/lihl/LLaFS2/data/sft_data_new_pe/sft_voc_dataset.json')
    df_all_with_description = pd.read_json('/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset_with_description.jsonl', lines=True, orient='records')
    df_base = df_all_with_description[df_all_with_description['class'].isin(base_classes)].reset_index(drop=True)
    df_novel = df_all_with_description[df_all_with_description['class'].isin(novel_classes)].reset_index(drop=True)

    overlap_classes = set(base_classes) & set(novel_classes)
    if overlap_classes:
        print(f"Warning: Overlapping classes found: {overlap_classes}")

    df_base.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/base_class_dataset_with_description.jsonl', orient='records', lines=True)
    df_novel.to_json('/data/lihl/LLaFS2/data/sft_data_without_attr/novel_class_dataset_with_description.jsonl', orient='records', lines=True)
    regular_dataset(data_path='/data/lihl/LLaFS2/data/sft_data_without_attr/base_class_dataset_with_description.jsonl', output_path='/data/lihl/LLaFS2/data/sft_data_new_pe/sft_voc_base.json')
    regular_dataset(data_path='/data/lihl/LLaFS2/data/sft_data_without_attr/novel_class_dataset_with_description.jsonl', output_path='/data/lihl/LLaFS2/data/sft_data_new_pe/sft_voc_novel.json')
    print(f'len of df_base: {len(df_base)}')
    print(f'len of df_novel: {len(df_novel)}')
    return all_class_dataset, base_class_dataset, novel_class_dataset

def regular_dataset(data_path, output_path):

    output_data = defaultdict(lambda: {"annotations": [], "image_size": None, "description": ""})
    
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            image_path = entry["image_path"]

            if not output_data[image_path]["description"]:
                output_data[image_path]["description"] = entry["description"]
            if not output_data[image_path]["image_size"]:
                output_data[image_path]["image_size"] = entry["image_size"]
            output_data[image_path]["annotations"].append({
                "category_name": entry["class"],
                "bbox": entry["annotation"]
            })
    converted_data = [
        {
            "image_path": key,
            "image_size": value["image_size"],
            "description": value["description"],
            "annotations": value["annotations"]
        }
        for key, value in output_data.items()
    ]
    
    # 写入 JSON 文件
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=4)

    print(f"数据已成功转换并保存到 {output_path}")

def convert_dataframe_to_json(df, output_file):
    output_data = defaultdict(lambda: {"annotations": [], "image_size": None, "description": ""})
    
    for _, row in df.iterrows():
        image_path = row["image_path"]
        if not output_data[image_path]["description"]:
            output_data[image_path]["description"] = row["description"]
        if not output_data[image_path]["image_size"]:
            output_data[image_path]["image_size"] = row["image_size"]
        output_data[image_path]["annotations"].append({
            "category_name": row["class"],
            "bbox": row["annotation"]
        })

    converted_data = [
        {
            "image_path": key,
            "image_size": value["image_size"],
            "description": value["description"],
            "annotations": value["annotations"]
        }
        for key, value in output_data.items()
    ]
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=4)
    print(f"数据已成功转换为 JSON 格式并保存到 {output_file}")

class Get_SFT_Dataset():
    def __init__(self, model_name, dataset, args=None, llava_args=None):
        self.feat_ext = feat_extractor()
        self.dataset = dataset
        self.dataset_v2 = dataset
        self.args = args
        print(f'Dataset length: {len(self.dataset_v2)}')
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
    
    def get_visual_tokens_v2(self, data):
        image = Image.open(data['image_path']).convert("RGB")
        caption = data['description']
        visual_tokens = self.feat_ext.forward(raw_img=image, caption=caption)
        return visual_tokens['multimodal_embeds'].detach().cpu().numpy()
        
    def prepare_dataset(self):
        #df = pd.read_json(self.dataset_path, lines=True)

        df = pd.DataFrame(self.dataset)
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
                instruction, input, output = get_prompt_new(
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

    def prepare_dataset_v3(self):
        """
        1. 将 self.dataset 按比例分成 训练集 / 验证集 / 测试集
        2. 分别处理三部分数据
        """
        # ---------- 1. 划分数据集 ----------
        random.seed(42)
        dataset = self.dataset[:]  # 拷贝一份，避免对原数据产生影响
        random.shuffle(dataset)

        total_samples = len(dataset)
        train_ratio, val_ratio = 0.8, 0.1
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        self.train_data = dataset[:train_end]
        self.val_data   = dataset[train_end:val_end]
        self.test_data  = dataset[val_end:]

        print(f"Dataset split completed:")
        print(f"  Train samples: {len(self.train_data)}")
        print(f"  Validation samples: {len(self.val_data)}")
        print(f"  Test samples: {len(self.test_data)}")

        # ---------- 2. 构建训练集的 sup_data_map ----------
        train_sup_data_map = build_sup_data_map(self.train_data)
        print(f"Train sup_data_map constructed with {len(train_sup_data_map)} categories.")

        # ---------- 3. 分别处理三部分数据 ----------
        self.process_single_dataset(self.train_data, "train", train_sup_data_map)
        self.process_single_dataset(self.val_data,   "val",   train_sup_data_map)
        self.process_single_dataset(self.test_data,  "test",  train_sup_data_map)


    def process_single_dataset(self, dataset, dataset_type, train_sup_data_map):
        """
        针对给定的 dataset（训练/验证/测试），做以下事情：
        1. 建立 sup_data_map，记录：cls_name -> [所有包含该类的 sup_data]
        2. 遍历每条 que_data，对其内部的 annotations 分组（同类的 bounding box 放在一起）。
        3. 对每个类别 cls_name 随机选取一个 sup_data，并把 sup_data 中不属于该类的注释过滤掉。
        4. 调用 get_prompt_new 构造 prompt，进行 tokenizer 和 label 构建。
        5. 批量保存（如果需要）。
        """
        dataset_processed = []
        data_dict_count = 0
        max_seq_length = getattr(self.args, "max_seq_length", 400)
        print(f"[{dataset_type}] Padding data, max_seq_length: {max_seq_length}")

        # ---------- (1) 建立 support 数据映射 ----------
        #sup_data_map = build_sup_data_map(dataset)

        # ---------- (2) 遍历每条数据 ----------
        for que_data in tqdm(dataset, desc=f"Processing {dataset_type} dataset"):
            # 先把 que_data 里的 bounding boxes 按照类名分组
            # 比如 { "bicycle": [anno1, anno2, anno3], "dog": [anno4], ... }
            category2annos = {}
            for ann in que_data["annotations"]:
                c = ann["category_name"]
                if c not in category2annos:
                    category2annos[c] = []
                category2annos[c].append(ann)

            # 针对这个 que_data 里的每个类别，各产出 1 条（query）数据
            for cls_name, annos_for_this_cls in category2annos.items():
                # 构造 query：只包含该类的所有 bounding boxes
                # （可能有 3 个 bicycle bbox，就一条数据里 annotations 有 3 个）
                que_data_single = {
                    "image_path":   que_data["image_path"],
                    "image_size":   que_data["image_size"],
                    "description":  que_data["description"],
                    "annotations":  annos_for_this_cls
                }

                # ---------- (3) 选取 sup_data ----------
                candidate_list = train_sup_data_map.get(cls_name, [])
                if not candidate_list:
                    # 如果没有任何支持数据包含此类，则跳过或做其它处理
                    print('Warning: No support data found! Please check your data.')
                    continue
                sup_data = random.choice(candidate_list)
                # 同样只保留 sup_data 里此类的 bbox
                sup_data_filtered = filter_data_for_class(sup_data, cls_name)

                # ---------- (4) 获取视觉特征 + 构造 prompt + tokenizer ----------
                sup_visual_tokens = torch.tensor(
                    self.get_visual_tokens_v2(sup_data_filtered)
                ).to("cuda")

                que_visual_tokens = torch.tensor(
                    self.get_visual_tokens_v2(que_data_single)
                ).to("cuda")

                # 注意这里要把 cls_name 也传给 get_prompt_new，方便区分
                instruction, user_input, output = get_prompt_new(
                    sup_data_filtered,
                    que_data_single,
                    cls_name
                )
                combined_text = instruction + user_input + output
                prompt_text = instruction + user_input

                # 分词
                tokenized = self.tokenizer(
                    combined_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length
                ).to("cuda")

                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]
                labels = input_ids.clone()

                # 对 prompt 部分的 label 设置 -100
                prompt_length = len(
                    self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
                )
                labels[:, :prompt_length] = -100

                # 对 pad_token_id 设置 -100
                labels[labels == self.tokenizer.pad_token_id] = -100
                labels[:, -1] = 128009  # 假设 128009 是 eos_token_id
                # 组合成单条数据
                item_dict = {
                    "input_ids":          input_ids,
                    "attention_mask":     attention_mask,
                    "labels":             labels,
                    "sup_visual_tokens":  sup_visual_tokens,
                    "que_visual_tokens":  que_visual_tokens
                }
                dataset_processed.append(item_dict)

                # 每达到一定数量就保存一次
                if len(dataset_processed) == 10000:
                    save_data_in_batches(
                        dataset_processed,
                        output_dir=f"{self.args.data_path}/{dataset_type}",
                        batch_size=len(dataset_processed),
                        start_batch_idx=data_dict_count
                    )
                    dataset_processed = []
                    data_dict_count += 1

        # ---------- (5) 保存剩余的 ----------
        if dataset_processed:
            save_data_in_batches(
                dataset_processed,
                output_dir=f"{self.args.data_path}/{dataset_type}",
                batch_size=len(dataset_processed),
                start_batch_idx=data_dict_count
            )
        print(f"[{dataset_type}] Dataset saved to {self.args.data_path}/{dataset_type}")

    def prepare_dataset_v2(self):
        dataset_processed = []
        data_dict_count = 0
        max_seq_length = self.args.max_seq_length if self.args.max_seq_length else 512
        print(f"Padding data, max_seq_length: {max_seq_length}")

        def choose_sup_data(current_que_data):
            while True:
                sup_data = random.choice(self.dataset)
                # 如果不想跟 query 是同一张图，就保证不选到同一个 image_path
                if sup_data["image_path"] != current_que_data["image_path"]:
                    return sup_data

        for idx, que_data in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Processing Data"):
            sup_data = choose_sup_data(que_data)
            
            sup_visual_tokens = torch.tensor(self.get_visual_tokens_v2(sup_data)).to("cuda")
            que_visual_tokens = torch.tensor(self.get_visual_tokens_v2(que_data)).to("cuda")

            instruction, user_input, output = get_prompt_new(sup_data, que_data)
            combined_text = instruction + user_input + output
            prompt_text = instruction + user_input  # 用于后面计算需要屏蔽 label 的位置

            tokenized = self.tokenizer(
                combined_text,
                return_tensors="pt",
                padding=False,       # 不直接在这里 pad
                truncation=True,     # 避免超过最大长度
                max_length=max_seq_length
            ).to("cuda")

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.clone()
            prompt_length = len(self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_length] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            item_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "sup_visual_tokens": sup_visual_tokens,
                "que_visual_tokens": que_visual_tokens
            }

            dataset_processed.append(item_dict)

            if len(dataset_processed) == 10000:
                save_data_in_batches(
                    dataset_processed,
                    output_dir=self.args.data_path,
                    batch_size=len(dataset_processed),
                    start_batch_idx=data_dict_count
                )
                dataset_processed = []
                data_dict_count += 1

        dataset_save_path = self.args.data_path
        save_data_in_batches(
            dataset_processed,
            output_dir=dataset_save_path,
            batch_size=len(dataset_processed),
            start_batch_idx=data_dict_count
        )
        print(f"Dataset saved to {dataset_save_path}")
    
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

if __name__ == '__main__':
    prepare_voc_dataset_without_attr('/data2/lihl/data/VOCdevkit/', split_id=1)
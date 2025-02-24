import torch
import os, ssl, glob
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig
from model.fsodllama import FSODLlamaForCausalLM
import xml.etree.ElementTree as ET

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def extract_info_from_xml(xml_path):
    """
    从PASCAL VOC格式的XML文件中提取图片信息和检测框数据。

    参数:
    xml_path (str): XML文件的路径。

    返回:
    dict: 包含图片信息和检测框数据的字典。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    bboxes = {}
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        bboxes[obj_name] = [(xmin, ymin), (xmax, ymax)]
        #bboxes.append((obj_name, xmin, ymin, xmax, ymax))
    
    info = {
        'filename': filename,
        'width': width,
        'height': height,
        'bboxes': bboxes
    }

    return info

def load_lora_model(
    model_path: str,
    model_base: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_name: str = "llama",
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str = "auto",
    device: str = "cpu",
    use_flash_attn: bool = False,
    **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}

    if model_base:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    if model_base:
        config = AutoConfig.from_pretrained(model_base)
        config.rope_scaling = None 
        model = FSODLlamaForCausalLM.from_pretrained(
            model_base,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        raise ValueError(f"No model found:{model_base}")
    
    special_tokens = ["<visual_sup>", "<visual_que>"]
    if not all(token in tokenizer.get_vocab() for token in special_tokens):
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        model.resize_token_embeddings(len(tokenizer))  # 调整嵌入层大小

    if model_path:
        print(f"Loading LoRA weights from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging weights...")
        model = model.merge_and_unload()
        print("Model loaded with LoRA weights.")

    model.eval()  # 设置为评估模式
    tokenizer = tokenizer
    model = model
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def load_ft_model(model_path='/data/lihl/llama-finetuned-all/checkpoint-good'):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    special_tokens = ["<visual_sup>", "<visual_que>"]
    if not all(token in tokenizer.get_vocab() for token in special_tokens):
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model = FSODLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    if len(special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    special_tokens = tokenizer.special_tokens_map
    #print(f"Special tokens: {special_tokens}")

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = model.to(device)
    print("sft Model Loaded!")
    return model, tokenizer

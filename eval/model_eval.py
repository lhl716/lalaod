import torch, random, json, re
import torch.nn as nn
from peft import PeftModel
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from data.util import load_data_from_batches
from model.fsodllama import FSODLlamaForCausalLM
from transformers import GenerationConfig, AutoConfig
import random
import numpy as np

def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Parameter name mismatch: {name1} vs {name2}")
            return False
        if not torch.equal(param1, param2):
            print(f"Mismatch found in parameter: {name1}")
            print(param1)
            print('-'*100)
            print(param2)
            print(f"param1 size: {param1.size()}, param2 size: {param2.size()}")
            return False
    print("All parameters are identical!")
    return True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pretrained_model(
    model_path: str,
    model_base: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map: str = "auto",
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
            device_map="cuda:0"
        )
        # model = FSODLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs) 
    else:
        raise ValueError(f"No Model Found: {model_base}")
    
    special_tokens = ["<visual_sup>", "<visual_que>"]
    if not all(token in tokenizer.get_vocab() for token in special_tokens):
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        model.resize_token_embeddings(len(tokenizer))  # 调整嵌入层大小

    if model_path:
        print(f"Loading LoRA weights from {model_path}")
        model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.bfloat16)
        #print("Merging weights...")
        #model = model.merge_and_unload()
        print("Model loaded with LoRA weights.")

    model.eval()  # 设置为评估模式
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

    special_tokens = tokenizer.special_tokens_map
    #print(f"Special tokens: {special_tokens}")

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = model.to(device)
    print("sft Model Loaded!")
    return model, tokenizer

def read_json(json_string):
    try:
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError:
        #print("Warning: Cannot parse the string as JSON, returning None instead.")
        return None

if __name__ == "__main__":
    model_base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "/data3/lihl/llama-lora-output/checkpoint-5131"
    model, tokenizer = load_pretrained_model(model_base=model_base, model_path=model_path)
    data_list = load_data_from_batches('/data3/lihl/sft_dataset_20250115/test')
    print(f'len(data_list): {len(data_list)}')
    num = random.randint(0,len(data_list))
    #num = 382, 2063
    #num = 19093, 17058
    num = 382
    print(f'选择的是第{num}张图片')
    input_ids = data_list[num]['input_ids']
    attention_mask = data_list[num]['attention_mask']
    sup_visual_tokens = data_list[num]['sup_visual_tokens']
    que_visual_tokens = data_list[num]['que_visual_tokens']

    # model, tokenizer = load_ft_model()
    eos_token_id = tokenizer.eos_token_id 
    
    model.eval()

    decoded_text = tokenizer.decode(input_ids[0])
    cleaned_text = re.sub(r"<\|start_header_id\|>assistant<\|end_header_id\|>.*", "", decoded_text, flags=re.DOTALL).strip()
    #print("Cleaned Text:\n", cleaned_text)
    # 正则表达式分组匹配
    pattern = r"(.*)(<\|start_header_id\|>assistant<\|end_header_id\|>).*"
    match = re.match(pattern, decoded_text, flags=re.DOTALL)
    print(decoded_text)

    if match:
        # 提取删除后的文本内容
        remaining_text = f"{match.group(1).strip()} {match.group(2).strip()}"  # 保留到 <|start_header_id|>assistant<|end_header_id|>
        removed_text = decoded_text[len(remaining_text):].strip()  # 删除后的部分
        #print("\nRemaining Text:\n", remaining_text)
        print("\nLabel:\n", removed_text)

    tokenized = tokenizer(
                    cleaned_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True
                ).to("cuda")

    after_input_ids = tokenized["input_ids"]
    after_input_ids = after_input_ids[:, 1:]

    #print("Re-encoded input_ids:\n", after_input_ids)
    #print(decoded_text)

    for i in trange(100):
        results = []
        with torch.no_grad():
            outputs = model.generate(
                #input_ids=inputs["input_ids"],
                #attention_mask=inputs["attention_mask"],
                inputs=after_input_ids,
                sup_visual_tokens=sup_visual_tokens,
                que_visual_tokens=que_visual_tokens,
                #max_length=512,
                max_new_tokens=128,
                do_sample=True,
                eos_token_id=eos_token_id,
                top_p=0.9,
                temperature=0.3
            )
            #print(outputs.size())
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            print(f'Generate text[0]: {generated_text[0]}')
            result = read_json(generated_text[0])
            results.append(result)
        break
    
    acc = 1 - results.count(None) / len(results)
    print(f'JSON达成率：{acc}')
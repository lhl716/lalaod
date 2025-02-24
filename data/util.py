import os, glob
from tqdm import trange
from safetensors.torch import save_file, load_file

def save_data_in_batches(data_list, batch_size, output_dir, start_batch_idx):
    """
    分批保存数据，每个文件保存 batch_size 条数据
    """
    os.makedirs(output_dir, exist_ok=True)
    # 确定起始文件编号
    batch_number = start_batch_idx
    for batch_idx in range(0, len(data_list), batch_size):
        batch_data = data_list[batch_idx:batch_idx + batch_size]
        save_dict = {}
        for i, sample in enumerate(batch_data):
            for key, value in sample.items():
                save_dict[f"sample_{i}_{key}"] = value
        # 保存当前批次文件
        save_file(save_dict, os.path.join(output_dir, f"batch_{batch_number}.safetensors"))
        batch_number += 1

def load_data_from_batches(input_dir):
    """
    从分批文件中加载数据，并恢复为每个样本一个字典的列表结构
    """
    file_paths = sorted(glob.glob(f"{input_dir}/*.safetensors"))
    data_list = []
    for file_path in file_paths:
        print(f"Loading {file_path}")
        batch_data = load_file(file_path)
        
        num_samples = len(set(k.split("_")[1] for k in batch_data.keys()))  # 根据 sample_x 推断样本数
        for i in trange(num_samples):
            sample = {}
            for key in batch_data.keys():
                if key.startswith(f"sample_{i}_"):
                    original_key = key.split("_", 2)[-1]
                    sample[original_key] = batch_data[key]
            data_list.append(sample)
            
            #break
        
    print(f"DATA LOADED!!! Len of train data: {len(data_list)}")
    return data_list

def load_data_for_eval(input_dir):
    """
    从分批文件中加载一条数据，并恢复为每个样本一个字典的列表结构
    """
    file_paths = sorted(glob.glob(f"{input_dir}/*.safetensors"))
    data_list = []
    for file_path in file_paths:
        print(f"Loading {file_path}")
        # 加载当前批次数据
        batch_data = load_file(file_path)
        
        # 恢复原始数据结构
        num_samples = len(set(k.split("_")[1] for k in batch_data.keys()))  # 根据 sample_x 推断样本数
        for i in range(num_samples):
            sample = {}
            for key in batch_data.keys():
                if key.startswith(f"sample_{i}_"):
                    original_key = key.split("_", 2)[-1]
                    sample[original_key] = batch_data[key]
            data_list.append(sample)

            break
        
    return data_list
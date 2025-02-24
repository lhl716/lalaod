import argparse, os, gc, json
import pandas as pd
from model.fsodllama import LLM_Model
from data.get_sft_dataset import Get_SFT_Dataset, prepare_voc_dataset_without_attr

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")

parser.add_argument('--model-name', type=str, default='llama3.1-8b', help='name of llm model like llama3.1-8b')
parser.add_argument('--processed-voc-data', type=str, default='/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset.jsonl', help="jsonl format file, include image_path, image_size, annotation, class")
parser.add_argument('--data-path', type=str, required=True, help='a folder path of dataset(train, val, test)')
parser.add_argument('--ft-mode', type=str, required=True, help='lora or full finetune(full_ft)')
parser.add_argument('--lora-rank', type=int, default=8, help='if you choose lora, you can give expected lora rank')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='training epoch')
parser.add_argument('--prepare-dataset', type=int, default=0, choices=[0, 1], help='prepare dataset? it will spend a long time. True or False, default False.')
parser.add_argument('--prepare-description', type=int, default=0, choices=[0, 1], help='need describe the image? True or False, default false.')
parser.add_argument('--vocdataset-path', type=str, default='/data2/lihl/data/VOCdevkit/', help='initial voc data path')
parser.add_argument('--max-seq-length', type=int, help='the max sequence length after tokenize the inputs')
parser.add_argument('--visual-proj-path', type=str, help='the visual projection embedding to project the visual tokens to  text embedding')
args = parser.parse_args()

def train(args):
    if args.prepare_dataset:
        split_id = 1
        #all_class_voc_path = os.path.join(os.path.dirname(args.data_path), 'all_class_dataset.jsonl')
        if os.path.exists(args.processed_voc_data):
            all_class_dataset = pd.read_json(args.processed_voc_data, lines=True)
            all_class_dataset = all_class_dataset.to_dict(orient='records')
        else:
            all_class_dataset, base_class_dataset, novel_class_dataset = prepare_voc_dataset_without_attr(args.vocdataset_path, split_id)
            print(f"Base class dataset size: {len(base_class_dataset)}")
            print(f"Novel class dataset size: {len(novel_class_dataset)}")
            print(f"All class dataset size: {len(all_class_dataset)}")
        print('Prepare dataset...')
        with open("/data/lihl/LLaFS2/data/sft_data_new_pe/sft_voc_dataset.json", "r") as f:
            data = json.load(f)
        Dataset = Get_SFT_Dataset(model_name=args.model_name, dataset=data, args=args)
        Dataset.prepare_dataset_v3()
        #Dataset.prepare_json_described_data()

        del Dataset
        gc.collect()

    llm = LLM_Model(args)
    if args.ft_mode == 'lora':
        llm.train()
    elif args.ft_mode == 'full_ft':
        llm.train_ft_all()

if __name__ =='__main__':
    train(args)
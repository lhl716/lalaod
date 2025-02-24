import torch, os
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from model.image_encoder import feat_extractor
from data.util import load_data_from_batches
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.modeling_fsodllama import DataCollatorForSupervisedDataset, PrepareDataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments,LlamaModel ,LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer

OUTPUT_DIR = "/data/lihl/fsod/llama-lora-output"

# 推理时为什么不能复现训练时的forward？
# 训练forward的输入和generate的输入是否一样？（prepare inputs for generation函数是否有问题？）
# 训练forward时候的采样策略是否和generate一样？（temperature + do_sample）

# ---------------------------------------------------General class-----------------------------------------------------
class FSODConfig(LlamaConfig):
    model_type = "FSOD_llama"

class FSODLlamaModel(LlamaModel):
    config_class = FSODConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

class FSODLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = FSODLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        loaded_model_name = kwargs.get("model_name")
        if loaded_model_name == 'llama3.1-8b':
            self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif loaded_model_name == 'llama2-7b':
            self.model_name = 'meta-llama/Llama-2-7b-chat-hf'
        else:
            print("No model name, loading llama3.1-8b")
            self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        #self.model_name = "codellama/CodeLlama-7b-hf"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        #self.projection = nn.Linear(768, self.model.config.hidden_size)
        #self.projection = self.projection.to(self.model.device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        覆盖父类方法：加载完主干后，再尝试加载 projection 层
        """
        model = super(FSODLlamaForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        visual_proj_path = kwargs.get("visual_proj_path","/data/lihl/fsod/model/visual_projection/visual_proj.pth")
        if os.path.exists(visual_proj_path):
            if not hasattr(model, "projection"):
                model.projection = nn.Linear(768, model.config.hidden_size, bias=False)
            model.projection.load_state_dict(torch.load(visual_proj_path))
            print(f"Projection layer loaded from {visual_proj_path}")
        else:
            if not hasattr(model, "projection"):
                model.projection = nn.Linear(768, model.config.hidden_size)
            print(f"Warning: No such file: {visual_proj_path}, visual projection initialing")
            model.projection = nn.Linear(768, model.model.config.hidden_size, bias=False)
            nn.init.orthogonal_(model.projection.weight)
            # torch.nn.init.xavier_uniform_(model.projection.weight)
            # 保存 projection 的参数
            torch.save(model.projection.state_dict(), visual_proj_path)
            print("Visual Projection parameters saved successfully!")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype  # 拿到模型主体第一块参数的 dtype
        model.projection = model.projection.to(device=device, dtype=dtype)
        
        return model

    def prepare_inputs(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            sup_visual_tokens, que_visual_tokens
    ):
        input_ids = input_ids.to(self.model.device)
        if labels is not None:
            labels = labels.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        _position_ids = position_ids

        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if isinstance(que_visual_tokens[0], torch.Tensor):
            #sup_visual_tokens = [tensor.cpu().tolist() for tensor in sup_visual_tokens]
            que_visual_tokens = [tensor.cpu().tolist() for tensor in que_visual_tokens]

        '''
        if isinstance(que_visual_tokens, torch.Tensor):
            sup_visual_tokens = self.projection(sup_visual_tokens.to(self.model.device).to(self.model.dtype))
        else:
            sup_visual_tokens = self.projection(torch.tensor(sup_visual_tokens, dtype=torch.float32).to(self.model.device).to(self.model.dtype))
        '''
        if isinstance(que_visual_tokens, torch.Tensor):
            que_visual_tokens = self.projection(que_visual_tokens.to(self.model.device).to(self.model.dtype))
        else:
            que_visual_tokens = torch.tensor(que_visual_tokens, dtype=self.model.dtype).to(self.model.device).to(self.model.dtype)
            que_visual_tokens = self.projection(que_visual_tokens)

        if "<visual_sup>" not in self.tokenizer.get_vocab():
            print("adding special token")
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_sup>', '<visual_que>']})
            self.model.resize_token_embeddings(len(self.tokenizer))  # 调整嵌入层

        sup_token_id = self.tokenizer.convert_tokens_to_ids("<visual_sup>")
        que_token_id = self.tokenizer.convert_tokens_to_ids("<visual_que>")

        if sup_token_id == self.tokenizer.unk_token_id or que_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Special tokens <visual_sup> or <visual_que> are not recognized by the tokenizer.")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).to("cuda")

        text_embeddings = self.model.embed_tokens(input_ids) 

        #sup_visual_tokens = sup_visual_tokens.to(self.model.device)
        que_visual_tokens = que_visual_tokens.to(self.model.device)
        text_embeddings = text_embeddings.to(self.model.device)
        assert isinstance(input_ids, torch.Tensor), f"Expected input_ids to be torch.Tensor, got {type(input_ids)}"
        sup_visual_index = (input_ids == self.tokenizer.convert_tokens_to_ids("<visual_sup>")).nonzero(as_tuple=True)
        que_visual_index = (input_ids == self.tokenizer.convert_tokens_to_ids("<visual_que>")).nonzero(as_tuple=True)

        #sup_visual_tokens = sup_visual_tokens.squeeze(1)
        que_visual_tokens = que_visual_tokens.squeeze(1)

        # 目前先把sup的token去掉了
        '''
        if len(sup_visual_index[0]) > 0:
            sup_start_idx = sup_visual_index[1][0].item()
            text_embeddings = torch.cat([
                text_embeddings[:, :sup_start_idx, :],
                sup_visual_tokens,  # [1, 32, embedding_dim]
                text_embeddings[:, sup_start_idx + 1:, :]
            ], dim=1)
            if attention_mask is not None:
                # 视觉token应不应该算loss？我觉得应该算
                attention_mask = torch.cat([
                    attention_mask[:, :sup_start_idx],
                    torch.ones((attention_mask.size(0), sup_visual_tokens.size(1))).to(self.model.device),  # 为视觉 token 添加注意力
                    attention_mask[:, sup_start_idx + 1:]
                ], dim=1)
            if labels is not None:
                labels = torch.cat([
                    labels[:, :sup_start_idx],
                    torch.full((labels.size(0), sup_visual_tokens.size(1)), -100, dtype=torch.long, device=labels.device),
                    labels[:, sup_start_idx + 1:]
                ], dim=1)
        '''

        if len(que_visual_index[0]) > 0:
            que_start_idx = que_visual_index[1][0].item()
            text_embeddings = torch.cat([
                text_embeddings[:, :que_start_idx, :],
                que_visual_tokens,  # [1, 32, embedding_dim]
                text_embeddings[:, que_start_idx + 1:, :]
            ], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask[:, :que_start_idx],
                    torch.ones((attention_mask.size(0), que_visual_tokens.size(1))).to(self.model.device),
                    attention_mask[:, que_start_idx + 1:]
                ], dim=1)
            if labels is not None:
                labels = torch.cat([
                    labels[:, :que_start_idx],
                    torch.full((labels.size(0), que_visual_tokens.size(1)), -100, dtype=torch.long, device=labels.device),
                    labels[:, que_start_idx + 1:]
                ], dim=1)

        new_input_embeds = text_embeddings.to(dtype=self.model.dtype)
        position_ids = torch.arange(0, text_embeddings.size(1), dtype=torch.long, device=text_embeddings.device).unsqueeze(0)
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, labels         

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            sup_visual_tokens: Optional[torch.FloatTensor] = None,
            que_visual_tokens: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            cache_position=None   
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                sup_visual_tokens,
                que_visual_tokens,
            )

        '''
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # 解码 logits 以生成预测文本
        if labels is not None and hasattr(self, 'tokenizer'):
            logits = outputs.logits  # [batch_size, seq_length, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_length]
            decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # 过滤掉 labels 中的 -100
            filtered_labels = labels.clone()
            filtered_labels[labels == -100] = self.tokenizer.pad_token_id  # 替换为 pad_token_id
            decoded_labels = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

            # 打印预测和目标文本
            for i, (pred, label) in enumerate(zip(decoded_predictions, decoded_labels)):
                print('-'*50, f"Sample {i}:", '-'*50)
                print('-'*50, "Predicted", '-'*50)
                print(pred)
                print('-'*50, 'Target', '-'*50)
                print(label)
        '''

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        sup_visual_tokens: Optional[torch.Tensor] = None,
        que_visual_tokens: Optional[torch.Tensor] = None, 
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if que_visual_tokens is not None:
            
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                sup_visual_tokens,
                que_visual_tokens,
            )
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            # sup_visual_tokens=sup_visual_tokens,
            # que_visual_tokens=que_visual_tokens,
            **kwargs
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        sup_visual_tokens = kwargs.pop("sup_visual_tokens", None)
        que_visual_tokens = kwargs.pop("que_visual_tokens", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if sup_visual_tokens is not None:
            inputs['sup_visual_tokens'] = sup_visual_tokens
        if que_visual_tokens is not None:
            inputs["que_visual_tokens"] = que_visual_tokens
        #print(f"input_ids:{input_ids}")
        return inputs
    
class LLM_Model():
    def __init__(self, args):
        self.args = args
        self.feat_ext = feat_extractor()
        if self.args.model_name == 'llama3.1-8b':
            self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif self.args.model_name == 'codellama-7b':
            self.model_name = "codellama/CodeLlama-7b-hf"
        elif self.args.model_name == "codellama-34b":
            self.model_name = 'codellama/CodeLlama-34b-hf'
        elif self.args.model_name == 'codellama-70b':
            self.model_name = 'codellama/CodeLlama-70b-hf'
        elif self.args.model_name == 'llama2-7b':
            self.model_name = 'meta-llama/Llama-2-7b-chat-hf'
        print('Loading LLM...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 设置 pad_token 为 eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 将 Namespace 转换为字典，因为下面读取FSODLlamaForCausalLM.from_pretrained的时候传入的是**kwargs
        # args_dict = vars(args)  

        config = AutoConfig.from_pretrained(self.model_name)
        config.rope_scaling = None 
        self.rank = args.lora_rank
        self.model = FSODLlamaForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **vars(args)
        )

        #for name, param in self.model.named_parameters():
        #    print(f"{name}: {param.device}")

        if "<visual_sup>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_sup>', '<visual_que>']})
            self.model.resize_token_embeddings(len(self.tokenizer))  # 调整模型的词汇表大小
        '''
        self.projection = nn.Linear(768, self.model.config.hidden_size)
        self.projection = self.projection.to(self.model.device)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        '''
        self.dataset = load_data_from_batches(args.data_path)
        
    def create_peft_config(self):
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_rank, #初始的是8
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
            #target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            #target_modules='all-linear',
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()       

    def train(self):
        self.create_peft_config()
        
        self.config = {
            "lora_config": self.peft_config,
            "learning_rate": self.args.lr,
            "num_train_epochs": self.args.epoch,
            "gradient_accumulation_steps": 4,
            "per_device_train_batch_size": 1,
            "gradient_checkpointing": False,
        }

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            bf16=True,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            optim="adamw_hf",
            max_steps=2416,
            dataloader_pin_memory=False,
            **{k: v for k, v in self.config.items() if k != "lora_config"},
        )

        train_dataset = PrepareDataset(
            #data_path = '/data/lihl/LLaFS2/data/sft_data_without_attr/training_batches_one_shot',
            data_path= self.args.data_path,
            tokenizer = self.tokenizer,
            data_args = None
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset, 
            data_collator=DataCollatorForSupervisedDataset(self.tokenizer),
            callbacks=[]
        )
        trainer.train()
        self.model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
    
    def train_ft_all(self):

        training_args = TrainingArguments(
            output_dir="llama-finetuned-all",
            num_train_epochs=1500,              # 实际调参时可以加大
            per_device_train_batch_size=1,   # 受限于显存，做个示例
            per_device_eval_batch_size=1,
            #evaluation_strategy="steps",     # 训练中定期评估
            #eval_steps=100,
            logging_steps=50,
            save_steps=200,
            #max_grad_norm=None, #如果出现ValueError: Attempting to unscale FP16 gradients.则关闭
            save_total_limit=2,             # 只保存最近的2个checkpoint
            #fp16=True,                      # 使用FP16混合精度
            bf16=True,
            learning_rate=2e-5,             # 全量微调时可适当调大，e.g. 2e-5 ~ 1e-4
            warmup_steps=100,
            gradient_accumulation_steps=4,   #可以根据需要设置
            report_to="none",               # 不使用 wandb 等日志平台时填 none
        )

        train_dataset = PrepareDataset(
            data_path = '/data/lihl/LLaFS2/data/sft_data_without_attr/training_batches_one_shot',
            tokenizer = self.tokenizer,
            data_args = None
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            data_collator=DataCollatorForSupervisedDataset(self.tokenizer),
        )

        trainer.train()

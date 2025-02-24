import torch, os
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union
from model.image_encoder import FeatExtractorWrapper
from data.util import load_data_from_batches
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.modeling_fsodllama import DataCollatorForSupervisedDataset, PrepareDataset, TimeRemainingCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments,LlamaModel ,LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer
from model.knowledge_base.kb_v4 import Knowledge_Base
from PIL import Image
from data.build_data.config import get_prompt_v2
from accelerate import Accelerator
accelerator = Accelerator()

OUTPUT_DIR = "/root/llama-lora-output"

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
        self.feat_ext = FeatExtractorWrapper(init_lavis=False)
        #self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        #self.projection = nn.Linear(768, self.model.config.hidden_size)
        #self.projection = self.projection.to(self.model.device)

        self.kb = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        覆盖父类方法：加载完主干后，再尝试加载 projection 层
        """
        model = super(FSODLlamaForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.feat_ext = FeatExtractorWrapper(init_lavis=True)
        visual_proj_path = kwargs.get("visual_proj_path","/root/fsod/model/visual_projection/visual_proj.pth")
        clip_visual_proj_path = kwargs.get("clip_visual_proj_path", "/root/fsod/model/visual_projection/clip_visual_proj.pth")
        visual_proj_path = "/root/fsod/model/visual_projection/visual_proj.pth"
        clip_visual_proj_path = "/root/fsod/model/visual_projection/clip_visual_proj.pth"
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

        if os.path.exists(clip_visual_proj_path):
            if not hasattr(model, "projection_clip"):
                model.projection_clip = nn.Linear(512, model.config.hidden_size, bias=False)
            model.projection_clip.load_state_dict(torch.load(clip_visual_proj_path))
            print(f"Clip Projection layer loaded from {clip_visual_proj_path}")
        else:
            if not hasattr(model, "projection_clip"):
                model.projection_clip = nn.Linear(512, model.config.hidden_size)
            print(f"Warning: No such file: {clip_visual_proj_path}, clip visual projection initialing")
            model.projection_clip = nn.Linear(512, model.model.config.hidden_size, bias=False)
            nn.init.orthogonal_(model.projection_clip.weight)
            # torch.nn.init.xavier_uniform_(model.projection.weight)
            # 保存 projection 的参数
            torch.save(model.projection_clip.state_dict(), clip_visual_proj_path)
            print("Clip Visual Projection parameters saved successfully!")

        model.projection = model.projection.to(device=model.device, dtype=model.dtype)
        model.projection_clip = model.projection_clip.to(device=model.device, dtype=model.dtype)


        # ==========【修改2：在这里初始化或加载 feat_ext】==========
        feat_ext_path = kwargs.get("feat_ext_path", "/root/.cache/torch/hub/checkpoints/blip2_pretrained.pth")
        if os.path.exists(feat_ext_path):
            # 如果用户指定的或者默认路径存在，则说明有已经保存好的特征提取器参数
            print(f"[from_pretrained] Found feat_ext_path: {feat_ext_path}, loading state_dict ...")
            # 构造空的 FeatExtractorWrapper(init_lavis=False)，再 load_state_dict
            model.feat_ext = FeatExtractorWrapper(init_lavis=True, device=model.device)
            model.feat_ext.to(device=model.device, dtype=model.dtype)
            # 加载已有权重（通常是你之前微调后保存的）
            loaded = torch.load(feat_ext_path, map_location=model.device)
            new_state = {}
            for k, v in loaded.items():
                if k.startswith("model."):
                    new_k = k.replace("model.", "blip2_model.", 1)
                else:
                    new_k = k
                new_state[new_k] = v

            model.feat_ext.load_state_dict(new_state, strict=False)
            #model.feat_ext.load_state_dict(state_dict)
            print(f"[from_pretrained] feat_ext loaded from {feat_ext_path}")
        else:
            # 如果文件不存在，则说明没有保存好的特征提取器参数
            # 那就用官方 LAVIS 预训练来初始化
            print(f"[from_pretrained] feat_ext_path not found ({feat_ext_path}), initializing from official LAVIS pretrained.")
            model.feat_ext = FeatExtractorWrapper(init_lavis=True, device=model.device)
            model.feat_ext.to(device=model.device, dtype=model.dtype)
            # 冻结/解冻Q-Former在 FeatExtractorWrapper 里已经写好
            # 可以选择立刻保存一份，避免下次还要去下载
            torch.save(model.feat_ext.state_dict(), feat_ext_path)
            print(f"[from_pretrained] FeatExtractorWrapper initialized and saved to {feat_ext_path}")
        # =======================================================
        
        return model
    
    def init_knowledge_base(self, kb_path=None, device="cpu"):
        #self.kb = Knowledge_Base(device=device)
        self.kb = Knowledge_Base.load("/root/fsod/model/knowledge_base/kb_pkl/KnowledgeBank.pkl", device=device)
        #self.kb.print_structure()

    def _insert_embeddings_between_special_tokens(
        self,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        insert_embeds: torch.Tensor,
        device: torch.device,
        mask_value: int = -100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        针对【单条样本】的 input_ids、text_embeddings、attention_mask、labels，
        在 (start_token_id, end_token_id) 之间插入 insert_embeds (形状 [M, hidden_dim])，
        并返回插入后的新 text_embeddings / attention_mask / labels（带回 [1, new_seq_len, ...] 或 [1, new_seq_len, hidden_dim] 的 batch 维度）。

        参数:
            input_ids:       [1, seq_len] 或 [seq_len]，表示单条样本的 token 序列
            text_embeddings: [1, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
            attention_mask:  [1, seq_len] 或 [seq_len]
            labels:          [1, seq_len] 或 [seq_len]
            start_token_id:  特殊 token 开始 ID
            end_token_id:    特殊 token 结束 ID
            insert_embeds:   [M, hidden_dim] 要插入的向量
            device:          torch.device
            mask_value:      在插入部分是否对 labels 使用 -100 (不计算 loss)，可按需修改

        返回:
            new_emb:  [1, new_seq_len, hidden_dim]
            new_am:   [1, new_seq_len]
            new_lb:   [1, new_seq_len]
        """

        # 1) 如果原先带 [1, seq_len, ...] 的 batch 维，就先 squeeze(0) 去掉
        #    以便后续操作更直观
        if input_ids.dim() == 2 and input_ids.size(0) == 1:
            input_ids = input_ids.squeeze(0)  # => [seq_len]
        if text_embeddings.dim() == 3 and text_embeddings.size(0) == 1:
            text_embeddings = text_embeddings.squeeze(0)  # => [seq_len, hidden_dim]
        if attention_mask.dim() == 2 and attention_mask.size(0) == 1:
            attention_mask = attention_mask.squeeze(0)  # => [seq_len]
        if labels.dim() == 2 and labels.size(0) == 1:
            labels = labels.squeeze(0)  # => [seq_len]

        # 如果插入向量为空，就直接把 batch 维补回来返回
        if insert_embeds is None or insert_embeds.size(0) == 0:
            return (
                text_embeddings.unsqueeze(0), 
                attention_mask.unsqueeze(0),
                labels.unsqueeze(0)
            )

        # 2) 查找 start_token_id / end_token_id 的位置
        start_positions = (input_ids == start_token_id).nonzero(as_tuple=True)
        end_positions = (input_ids == end_token_id).nonzero(as_tuple=True)

        # 如果找不到对应区间，就直接返回原始结果并加回 batch 维
        if len(start_positions[0]) == 0 or len(end_positions[0]) == 0:
            return (
                text_embeddings.unsqueeze(0),
                attention_mask.unsqueeze(0),
                labels.unsqueeze(0),
            )

        start_idx = start_positions[0][0].item()
        end_idx   = end_positions[0][0].item()

        # 3) 对 text_embeddings 做切片拼接
        left_part  = text_embeddings[: start_idx + 1, :]  # [start_idx+1, hidden_dim]
        right_part = text_embeddings[end_idx:, :]         # [seq_len-end_idx, hidden_dim]
        
        # 组合 new_emb = left + insert_embeds + right
        new_emb = torch.cat([left_part, insert_embeds, right_part], dim=0)  
        # => [ ( (start_idx+1) + M + (seq_len-end_idx) ), hidden_dim ]

        # 4) 同步修改 attention_mask
        left_am   = attention_mask[: start_idx + 1]  # [start_idx+1]
        right_am  = attention_mask[end_idx:]         # [seq_len-end_idx]
        mid_am    = torch.ones(insert_embeds.size(0), dtype=left_am.dtype, device=self.model.device)
        new_am = torch.cat([left_am, mid_am, right_am], dim=0)
        # => [ (start_idx+1) + M + (seq_len-end_idx) ]
        # 5) 同步修改 labels
        left_lb  = labels[: start_idx + 1]
        right_lb = labels[end_idx:]
        mid_lb   = torch.full(
            (insert_embeds.size(0),),
            mask_value,  # 通常 -100，表示不计算 loss
            dtype=labels.dtype,
            device=self.model.device
        )
        new_lb = torch.cat([left_lb, mid_lb, right_lb], dim=0)
        # => [ (start_idx+1) + M + (seq_len-end_idx) ]
        # 6) 恢复到 [1, new_seq_len, ...]（batch 维度 = 1）
        # new_emb = new_emb.unsqueeze(0)  # => [1, new_seq_len, hidden_dim]
        # new_am  = new_am.unsqueeze(0)   # => [1, new_seq_len]
        # new_lb  = new_lb.unsqueeze(0)   # => [1, new_seq_len]

        return new_emb, new_am, new_lb

    def prepare_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        image_path: Optional[list] = None,
        caption: Optional[list] = None,
        class_name: Optional[list] = None,
        annotations: Optional[list] = None,
        image_size: Optional[list] = None
    ):
        batchsize = len(image_path)
        batch_attention_mask = []
        batch_labels = []
        batch_input_embeds = []
        batch_position_ids = []
        _position_ids = position_ids
        if "<visual_sup_start>" not in self.tokenizer.get_vocab():
            new_special_tokens = [
                "<visual_sup_start>", "<visual_sup_end>",
                "<visual_que_start>", "<visual_que_end>",
                "<attr_embeds_start>", "<attr_embeds_end>"
            ]
            print("adding special token")
            self.tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))
        # 如果仅是单步推理（如generate的step） input_ids.shape 可能是 [batch_size, 1]
        if input_ids is not None and input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        for idx in range(batchsize):
            instruction, input, output = get_prompt_v2(
                class_name=class_name[idx], image_size=image_size[idx], annotations=annotations[idx], caption=caption[idx]
            )
            combined_text = instruction + input + output
            prompt_text = instruction + input
            tokenized = self.tokenizer(
                combined_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024
            ).to(self.model.device)

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.clone()
            # 对 pad_token_id 设置 -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[:, -1] = 128009  # 假设 128009 是 eos_token_id

            # 对 prompt 部分的 label 设置 -100
            prompt_length = len(
                self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
            )
            labels[:, :prompt_length] = -100
            filtered_labels = labels.clone()
            filtered_labels[labels == -100] = self.tokenizer.pad_token_id  # 替换为 pad_token_id
            decoded_labels = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            '''
            input_ids = input_ids.to(self.model.device)
            if labels is not None:
                labels = labels.to(self.model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            '''
            # ---- 1) 如果有图像路径，进行特征提取 + 知识库查询 ---- #
            sup_visual_tokens = None
            que_visual_tokens = None
            all_attr_embed    = None  # 你可能还有第三种embedding

            if image_path is not None:
                raw_img = Image.open(image_path[idx]).convert("RGB")
                que_visual_tokens = self.feat_ext.forward(raw_img=raw_img, caption=caption[idx])
                que_visual_tokens = que_visual_tokens['multimodal_embeds']
                all_attr_embed_, refer_feature_map_ = self.kb.generate_refer_feature_map(raw_img)
                sup_visual_tokens = self.projection_clip(refer_feature_map_.to(self.model.device, dtype=self.model.dtype)).to(self.model.device)
                que_visual_tokens = self.projection(que_visual_tokens.to(self.model.device, dtype=self.model.dtype)).to(self.model.device)
                que_visual_tokens = que_visual_tokens.squeeze(0)
                all_attr_embed_ = [x.to(self.model.device, dtype=self.model.dtype) for x in all_attr_embed_]
                all_attr_embed_ = torch.stack(all_attr_embed_, dim=0)  # 在dim=0堆叠
                all_attr_embed  = self.projection_clip(all_attr_embed_).to(self.model.device)

            # ---- 2)并获取 token_id ---- #
            sup_start_id  = self.tokenizer.convert_tokens_to_ids("<visual_sup_start>")
            sup_end_id    = self.tokenizer.convert_tokens_to_ids("<visual_sup_end>")
            que_start_id  = self.tokenizer.convert_tokens_to_ids("<visual_que_start>")
            que_end_id    = self.tokenizer.convert_tokens_to_ids("<visual_que_end>")
            attr_start_id = self.tokenizer.convert_tokens_to_ids("<attr_embeds_start>")
            attr_end_id   = self.tokenizer.convert_tokens_to_ids("<attr_embeds_end>")

            # ---- 3) 先做 text_embeddings ---- #
            text_embeddings = self.model.embed_tokens(input_ids).to(self.model.device)
            # ---- 4) 分别插入 sup/que/all_attr embeds ---- #
            if sup_visual_tokens is not None:
                text_embeddings, attention_mask, labels = self._insert_embeddings_between_special_tokens(
                    input_ids=input_ids,
                    text_embeddings=text_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    start_token_id=sup_start_id,
                    end_token_id=sup_end_id,
                    insert_embeds=sup_visual_tokens,  # [M, hidden_dim]
                    device=self.model.device,
                    mask_value=-100
                )

            if que_visual_tokens is not None:
                text_embeddings, attention_mask, labels = self._insert_embeddings_between_special_tokens(
                    input_ids=input_ids,
                    text_embeddings=text_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    start_token_id=que_start_id,
                    end_token_id=que_end_id,
                    insert_embeds=que_visual_tokens,  # [M, hidden_dim]
                    device=self.model.device,
                    mask_value=-100
                )

            # 如果还有 all_attr_embed:
            if all_attr_embed is not None and all_attr_embed.size(0) > 0:
                text_embeddings, attention_mask, labels = self._insert_embeddings_between_special_tokens(
                    input_ids=input_ids,
                    text_embeddings=text_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    start_token_id=attr_start_id,
                    end_token_id=attr_end_id,
                    insert_embeds=all_attr_embed,
                    device=self.model.device,
                    mask_value=-100
                )

            # ---- 5) 生成新的 input_embeds, position_ids ---- #
            new_input_embeds = text_embeddings.to(dtype=self.model.dtype)

            if position_ids is not None:
                position_ids = position_ids.to(self.model.device)
            else:
                seq_length = new_input_embeds.size(0)
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.model.device).unsqueeze(0)
            position_ids = position_ids.squeeze(0)
            
            batch_input_embeds.append(new_input_embeds)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_position_ids.append(position_ids)
            position_ids = _position_ids

        pad_embed = self.model.get_input_embeddings()(
            torch.tensor([self.tokenizer.pad_token_id], device=self.model.device)
        ).squeeze(0)  # [hidden_size]
        processed_batch = []
        for emb in batch_input_embeds:
            processed_batch.append(emb.to(dtype=self.model.dtype)) 

        def custom_pad_sequence(sequences, pad_value):
            max_len = max(s.size(0) for s in sequences)
            out_tensors = []
            for tensor in sequences:
                padding_size = max_len - tensor.size(0)
                if padding_size > 0:
                    padding = pad_value.repeat(padding_size, 1)  # [pad_len, hidden]
                    padded = torch.cat([tensor, padding], dim=0)
                else:
                    padded = tensor
                out_tensors.append(padded)
            return torch.stack(out_tensors, dim=0)  # [batch, seq, hidden]

        new_input_embeds = custom_pad_sequence(processed_batch, pad_embed)
        #new_input_embeds = torch.nn.utils.rnn.pad_sequence(batch_input_embeds, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.model.device)
        new_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100).to(self.model.device)
        new_position_ids = torch.nn.utils.rnn.pad_sequence(batch_position_ids, batch_first=True, padding_value=-100).to(self.model.device)
        return None, new_position_ids, new_attention_mask, past_key_values, new_input_embeds, new_labels
    
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
            return_dict: Optional[bool] = None,
            image_path: Optional[str] = None,
            caption: Optional[str] = None,
            class_name: Optional[str] = None,
            annotations: Optional[str] = None,
            image_size: Optional[list] = None,
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
                image_path=image_path,
                caption=caption,
                class_name=class_name,
                annotations=annotations,
                image_size=image_size
            )

        
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
        device = next(self.parameters()).device
        if isinstance(outputs, CausalLMOutputWithPast):
            outputs = CausalLMOutputWithPast(
                loss=outputs.loss.to(device) if outputs.loss is not None else None,
                logits=outputs.logits.to(device),
                past_key_values=outputs.past_key_values,
                hidden_states=[x.to(device) for x in outputs.hidden_states] if outputs.hidden_states is not None else None,
                attentions=[x.to(device) for x in outputs.attentions] if outputs.attentions is not None else None
            )
        
        '''
        # 解码 logits 以生成预测文本
        if labels is not None and hasattr(self, 'tokenizer'):
            logits = outputs.logits  # [batch_size, seq_length, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_length]
            decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            print(labels)
            filtered_labels = labels.clone()
            filtered_labels[labels == -100] = self.tokenizer.pad_token_id  # 替换为 pad_token_id
            decoded_labels = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

            for i, (pred, label) in enumerate(zip(decoded_predictions, decoded_labels)):
                print('-'*50, f"Sample {i}:", '-'*50)
                print('-'*50, "Predicted", '-'*50)
                print(pred)
                print('-'*50, 'Target', '-'*50)
                print(label)

        print("\n" + "="*40 + " Device Check " + "="*40)
        args_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict
        }
        
        def print_device_info(name, value, indent=0):
            prefix = " " * indent
            if value is None:
                print(f"{prefix}{name}: None")
                return
            if isinstance(value, torch.Tensor):
                print(f"{prefix}{name}: {value.device} | shape: {value.shape} | dtype: {value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}{name}:")
                for i, item in enumerate(value):
                    print_device_info(f"[{i}]", item, indent+4)
            elif isinstance(value, dict):
                print(f"{prefix}{name}:")
                for k, v in value.items():
                    print_device_info(k, v, indent+4)
            else:
                print(f"{prefix}{name}: {type(value)}")

        for name, value in args_dict.items():
            print_device_info(name, value)
        
        print("="*90 + "\n")
        '''
        
        return outputs
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
        image_path: Optional[str] = None, 
        caption: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if image_path is not None:
            
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
                image_path=image_path,
                caption=caption
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
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # 若在生成时依然想带上 image_path，可以这样透传
        if 'image_path' in kwargs:
            inputs['image_path'] = kwargs['image_path']
        if 'caption' in kwargs:
            inputs['caption'] = kwargs['caption']
        return inputs
    
class LLM_Model():
    def __init__(self, args):
        self.args = args
        #self.feat_ext = feat_extractor()
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
            pretrained_model_name_or_path=self.model_name,
            config=config,
            device_map = "auto",
            **vars(args)
        )

        #for name, param in self.model.named_parameters():
        #    print(f"{name}: {param.device}")


        '''
        self.projection = nn.Linear(768, self.model.config.hidden_size)
        self.projection = self.projection.to(self.model.device)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        '''
        #self.dataset = load_data_from_batches(args.data_path)
        
    def create_peft_config(self):
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_rank, #初始的是8
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        for name, param in self.model.named_parameters():
            if "Qformer" in name:
                param.requires_grad = True
        self.model.print_trainable_parameters()       
        
    def train(self):
        if "<visual_sup_start>" not in self.tokenizer.get_vocab():
            new_special_tokens = [
                "<visual_sup_start>", "<visual_sup_end>",
                "<visual_que_start>", "<visual_que_end>",
                "<attr_embeds_start>", "<attr_embeds_end>"
            ]
            self.tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.init_knowledge_base()

        for n, m in self.model.named_modules(): print(n)
        
        self.create_peft_config()
        
        self.config = {
            "lora_config": self.peft_config,
            "learning_rate": self.args.lr,
            "num_train_epochs": self.args.epoch,
            "gradient_accumulation_steps": 4,
            "per_device_train_batch_size": 2,
            "gradient_checkpointing": False,
        }
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            bf16=True,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=3,
            optim="adamw_hf",
            dataloader_pin_memory=False,
            # 可以根据需求开启评估策略: epoch 或 steps
            #eval_strategy="epoch",   # "steps" / "epoch" / "no"
            # 如果 evaluation_strategy="steps"，可以再设置 eval_steps
            # eval_steps=100,
            **{k: v for k, v in self.config.items() if k != "lora_config"},
        )

        train_data_path = os.path.join(self.args.data_path, 'train')
        train_dataset = PrepareDataset(
            data_path= "/root/fsod/data/coco/coco_train.jsonl",
            tokenizer = self.tokenizer,
            data_args = None
        )
        print(f'Using {train_data_path} as train dataset')
        '''
        val_dataset = None
        val_data_path = os.path.join(self.args.data_path, 'val')  # 假设验证数据文件夹名为 "val"
        if os.path.exists(val_data_path):
            val_dataset = PrepareDataset(
                data_path= val_data_path,
                tokenizer = self.tokenizer,
                data_args = None
            )
            print(f'Using {val_data_path} as validation dataset')
        '''

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset, 
            #eval_dataset=val_dataset,
            data_collator=DataCollatorForSupervisedDataset(self.tokenizer),
            callbacks=[TimeRemainingCallback()]
        )
        with open("/root/fsod/model/output.txt", "w") as f:
            for name, param in self.model.named_parameters():
                device = param.device
                requires_grad = param.requires_grad
                f.write(f"Layer: {name}, Device: {device}, Requires Grad: {requires_grad}\n")
        #trainer = accelerator.prepare(trainer)
        trainer.train()
        self.model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
    
    def train_ft_all(self):

        training_args = TrainingArguments(
            output_dir="/root/llama-finetuned-all",
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
            data_path = '/root/LLaFS2/data/sft_data_without_attr/training_batches_one_shot',
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

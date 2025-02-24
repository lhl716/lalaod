import torch, os
import torch.nn as nn
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel

class feat_extractor():
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", torch.device("cuda") if torch.cuda.is_available() else 'cpu')
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name='blip2_feature_extractor', 
            model_type='pretrain', 
            is_eval=True, 
            device=self.device
        )

        for name, param in self.model.named_parameters():
            if "Qformer" in name:
                param.requires_grad = True  # 冻结视觉编码器
            else:
                param.requires_grad = False  # 微调 Q-Former 和投影层
        
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        #print("特征提取器可训练参数层:", trainable_params)
        print("特征提取器可训练参数数量:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        
    def forward(self, raw_img, caption):
        image = self.vis_processors["eval"](raw_img).unsqueeze(0).to(self.device)
        text_input = self.txt_processors["eval"](caption)
        sample = {"image": image, "text_input":[text_input]}
        features_multimodal = self.model.extract_features(sample)
        return features_multimodal
    
    def print_arch(self):
        print("Model Architecture:")
        for name, module in self.model.named_children():
            print(f'Layer: {name}')
        print(self.model.visual_encoder)

import torch
import torch.nn as nn
from lavis.models import load_model_and_preprocess

class FeatExtractorWrapper(nn.Module):
    """
    将原先的 feat_extractor 改造成 PyTorch 子模块。
    通过 init_lavis 控制是否直接用 LAVIS 官方权重初始化。
    """
    def __init__(self, init_lavis=True, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        if init_lavis:
            # 如果 init_lavis=True，则从官方预训练加载 BLIP2 Feature Extractor
            print("[FeatExtractorWrapper] Initializing from official LAVIS pretrained (online or cache).")
            self.blip2_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name='blip2_feature_extractor',
                model_type='pretrain',
                is_eval=False,
                device=self.device
            )
        else:
            # 如果 init_lavis=False，就先占位
            print("[FeatExtractorWrapper] Creating empty BLIP2 wrapper (will load state_dict externally).")
            self.blip2_model = None

    def load_pretrained_model(self):
        """
        如果 init_lavis=False 时，可以在外部 load_state_dict 之前，
        需要先手动再去构造 blip2_model 结构。
        不过如果你直接 load_state_dict 到已有结构也行。
        """
        # 按需实现。这里示例略过。
        pass

    def forward(self, raw_img, caption: str):
        if self.blip2_model is None:
            raise ValueError("blip2_model is not initialized. Load or init_lavis=True first.")
        image = self.vis_processors["eval"](raw_img).unsqueeze(0).to(self.device)
        text_input = self.txt_processors["eval"](caption)
        sample = {"image": image, "text_input":[text_input]}
        features_multimodal = self.blip2_model.extract_features(sample)
        return features_multimodal
    
class Self_Attention():
    def __init__(self, **kwargs):
        pass

class Cross_Attention():
    def __init__(self, **kwargs):
        pass

class FFW():
    def __init__(self, **kwargs):
        pass

class AttrEmbedding():
    def __init__(self, **kwargs):
        pass

class MultimodalEncoder():
    def __init__(self, **kwargs):
        pass

if __name__ == '__main__':
    feat_ext = feat_extractor()
    image = Image.open('/data2/lihl/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg').convert("RGB")
    caption = "A room with white and red wallpaper, a hanging mirror and a clock that are all made in wood."
    visual_tokens = feat_ext.forward(raw_img=image, caption=caption)
    visual_tokens = visual_tokens['multimodal_embeds']
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
    
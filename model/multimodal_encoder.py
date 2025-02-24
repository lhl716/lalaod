import torch, os
import torch.nn as nn
import numpy as np
from PIL import Image

from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel

class feat_extractor():
    """
    已有的特征提取器，用于通过 BLIP-2 中的 Q-Former
    得到图像的特征 (32, 768)。
    """
    def __init__(self, **kwargs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name='blip2_feature_extractor', 
            model_type='pretrain', 
            is_eval=True, 
            device=self.device
        )
        
    def forward(self, raw_img, caption):
        image = self.vis_processors["eval"](raw_img).unsqueeze(0).to(self.device)
        text_input = self.txt_processors["eval"](caption)
        sample = {"image": image, "text_input":[text_input]}
        features_multimodal = self.model.extract_features(sample)
        # features_multimodal["image_embeds_proj"] 通常是 (B, 32, 768)
        # 你可以视自己的需求，具体取用哪个输出
        return features_multimodal["multimodal_embeds"]


class Self_Attention(nn.Module):
    """
    用最简单的MultiHeadAttention来做自注意力。
    这里演示一个最基础的写法。
    """
    def __init__(self, embed_dim=768, num_heads=8, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, 
                                         num_heads=num_heads,
                                         batch_first=True)
        
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        attn_output, attn_weights = self.mha(query=x, key=x, value=x)
        # 这里也可以加上残差连接和LayerNorm
        out = attn_output
        return out


class Cross_Attention(nn.Module):
    """
    用最简单的MultiHeadAttention来做Cross Attention。
    query 来自 x，key/value 来自 context。
    """
    def __init__(self, embed_dim=768, num_heads=8, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, 
                                         num_heads=num_heads,
                                         batch_first=True)

    def forward(self, x, context):
        # x: (batch_size, seq_len_x, embed_dim)
        # context: (batch_size, seq_len_ctx, embed_dim)
        attn_output, attn_weights = self.mha(query=x, 
                                             key=context, 
                                             value=context)
        # 同样可加残差+LayerNorm
        out = attn_output
        return out


class FFW(nn.Module):
    """
    一个简易的前馈网络(Feed Forward)，可根据需要再加Dropout等。
    """
    def __init__(self, embed_dim=768, hidden_dim=1024, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class AttrEmbedding(nn.Module):
    """
    属性图像编码器，使用CLIP来提取属性图片的向量表征 (batch, 512)。
    可根据需求把输出再投影到和Q-Former一致的 768 维。
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, **kwargs):
        super().__init__()
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        # 如果需要把 512 -> 768，可启用以下投影层
        self.proj = nn.Linear(512, 768)
        
    def forward(self, attr_imgs):
        """
        attr_imgs: 一个包含若干PIL.Image或tensor图像的list
        返回: (batch_size, num_attrs, embed_dim=768)
        """
        inputs = self.processor(images=attr_imgs, return_tensors="pt").to(self.device)
        image_embeds = self.model.get_image_features(**inputs)
        # image_embeds: (batch_size * num_imgs, 512)
        # 这里简单假设 batch_size=1，把所有属性图像拼成一个批次
        image_embeds = self.proj(image_embeds)  # (batch_size * num_imgs, 768)
        # 例如 batch_size=1, num_attrs = len(attr_imgs)
        # 可以 reshape 为 (1, num_attrs, 768)
        image_embeds = image_embeds.unsqueeze(0)  # (1, num_attrs, 768)
        return image_embeds


class MultimodalEncoder(nn.Module):
    """
    主体的多模态编码流程：
    1) 从外部输入 Q-Former 的图像特征 image_feats: (B, 32, 768)
    2) 对文本进行编码 (使用CLIP)，得到文本特征 text_feats: (B, T, 512) -> 再投影到768
    3) 拼接后通过自注意力 + 前馈网络 -> o1
    4) 属性图像特征 attr_feats: (B, N, 768)
    5) o1 和 attr_feats 做 Cross Attention -> o2
    6) o2 通过前馈网络 -> final
    """
    def __init__(self, 
                 text_model_name="openai/clip-vit-base-patch32",
                 embed_dim=768,
                 text_hidden_dim=512, 
                 num_heads=8,
                 device=None, 
                 **kwargs):
        super().__init__()
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else 'cpu')

        # 文本编码器 (CLIP)
        self.text_processor = CLIPProcessor.from_pretrained(text_model_name)
        self.text_model = CLIPModel.from_pretrained(text_model_name).to(self.device)
        
        # 投影层，把 CLIP 文本特征 (512) -> (768)，
        # 如果 CLIP 模型输出特征已经是 768，可以视情况省略
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim)

        # 自注意力和跨注意力
        self.self_attn = Self_Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn = Cross_Attention(embed_dim=embed_dim, num_heads=num_heads)

        # 前馈网络
        self.ffw1 = FFW(embed_dim=embed_dim, hidden_dim=embed_dim*4)
        self.ffw2 = FFW(embed_dim=embed_dim, hidden_dim=embed_dim*4)

    def encode_text(self, texts):
        """
        使用CLIP对文本进行编码，返回投影后 (B, T, 768) 形状的特征。
        texts: list of str
        """
        inputs = self.text_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_embeds = self.text_model.get_text_features(**inputs)  # (B, 512)
        # CLIP默认只返回[CLS]的 global 语义，若需要token级输出需使用 text_model.forward(...)
        text_embeds = self.text_proj(text_embeds)  # (B, 768)
        # 为了后续与 image_feats 进行拼接，需要把 text_embeds 增加一个 seq 维度
        text_embeds = text_embeds.unsqueeze(1)  # (B, 1, 768)，此处仅示例
        return text_embeds

    def forward(self, image_feats, texts, attr_feats):
        """
        image_feats: (B, 32, 768) Q-Former 输出
        texts: list of str
        attr_feats: (B, N, 768) 属性图像特征(假设已经通过 AttrEmbedding 得到了 768 维)
        """
        # 1) 文本编码
        text_embeds = self.encode_text(texts)  # (B, 1, 768)，示例只用 global embedding

        # 2) 拼接图像特征和文本特征
        # 例如在 seq 维度进行拼接
        x = torch.cat([image_feats, text_embeds], dim=1)  # (B, 32+1, 768)

        # 3) 自注意力
        x = self.self_attn(x)    # (B, 33, 768)
        x = self.ffw1(x)         # (B, 33, 768)
        o1 = x                   # 命名为 o1

        # 4) Cross Attention (o1和属性特征)
        #   这里 query = o1, key/value = attr_feats
        o2 = self.cross_attn(o1, attr_feats)  # (B, 33, 768)
        
        # 5) 前馈网络
        out = self.ffw2(o2)      # (B, 33, 768)
        return out


# =============================
# 使用示例
# =============================
if __name__ == "__main__":
    # 假设我们已经有一张PIL图像和一个属性图像列表:
    raw_img = Image.new('RGB', (224, 224), color='red')
    attr_imgs = [Image.new('RGB', (224, 224), color='green'),
                 Image.new('RGB', (224, 224), color='blue')]

    # 1) 得到 Q-Former 的图像特征 (B, 32, 768)
    extractor = feat_extractor()
    qformer_feats = extractor.forward(raw_img, "A red image")  # (1, 32, 768)

    # 2) 得到属性图像特征 (1, N, 768)
    attr_embedding = AttrEmbedding()
    attr_feats = attr_embedding(attr_imgs)  # (1, len(attr_imgs), 768)

    # 3) 构建多模态编码器
    mm_encoder = MultimodalEncoder()

    # 4) 前向计算
    #    texts 可以是一个list，例如 ["this is a caption"]
    final_output = mm_encoder(qformer_feats, ["this is a caption"], attr_feats)

    print("Final multimodal feature shape:", final_output.shape)
    # 预期输出 (1, 33, 768)  (若拼接后为 32+1，再与属性做CrossAttn)
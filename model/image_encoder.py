import torch, os
import torch.nn as nn
from lavis.models import load_model_and_preprocess

class feat_extractor():
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
        return features_multimodal

class Img_projection():
    def __init__(self, **kwargs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        visual_proj_path = kwargs.get("visual_proj_path","/data/lihl/fsod/model/visual_projection/visual_proj.pth")
        self.projection = nn.Linear(768, self.model.config.hidden_size)
        
        if os.path.exists(visual_proj_path):
            self.projection.load_state_dict(torch.load(visual_proj_path))
            print(f"Visual Projection loaded {visual_proj_path}")
        else:
            print(f"Warning: No such file: {visual_proj_path}, visual projection initialing")
            self.projection = nn.Linear(768, self.model.config.hidden_size)
            torch.nn.init.xavier_uniform_(self.projection.weight)
            # 保存 projection 的参数
            torch.save(self.projection.state_dict(), visual_proj_path)
            print("Visual Projection parameters saved successfully!")
        self.projection = self.projection.to(self.model.device)
    
    def forward(self, img_token):
        return self.projection(img_token)
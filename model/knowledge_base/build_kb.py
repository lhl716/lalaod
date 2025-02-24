import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from model.knowledge_base.util import VOCDataLoader, AttributeGenerator, AttributeManager
from kb_v4 import Knowledge_Base

class KnowledgeBaseBuilder:
    def __init__(self, split_id, voc_root, openai_key=None):
        self.split_id = split_id
        self.voc_loader = VOCDataLoader(voc_root, split_id)
        
        # 初始化属性系统
        self.attr_manager = AttributeManager(split_id)
        #self.attr_generator = AttributeGenerator(openai_key) if openai_key else None
        
        # 知识库实例
        self.kb = Knowledge_Base(
            segmentation_method="SLIC",
            similarity_threshold=0.26
        )

    def build(self, force_regenerate=False):
        # 获取当前split的所有base classes
        base_classes = self.voc_loader.get_base_classes()
        all_classes = self.voc_loader.get_all_classes()
        
        # 步骤1: 生成/加载属性描述
        if force_regenerate or not self.attr_manager.attributes:
            if not self.attr_generator:
                raise ValueError("API key required for initial generation")
                
            #self.attr_manager.generate_all(base_classes, self.attr_generator)
            self.attr_manager.generate_all(all_classes, self.attr_generator)
        
        # 步骤2: 初始化知识库文本特征
        for cls in base_classes:
            attr_text = self.attr_manager.get_attribute(cls)
            if not attr_text:
                print(f"Warning: Missing attributes for {cls}")
                continue
            self.kb.process_attributes(attr_text)

        # 步骤3: 处理图像数据（与原流程相同）
        self._process_images(base_classes)
        
        return self.kb

    def _process_images(self, base_classes):
        image_list = self.voc_loader.get_image_list()
        valid_count = 0
        missing_mask_count = 0
        for img_id in tqdm(image_list, desc="Processing images"):
            # 加载图像和语义分割掩码（P模式）
            image = self.voc_loader.load_image(img_id)
            xml_path = self.voc_loader.get_annotation_path(img_id)
            
            if not xml_path:
                missing_mask_count += 1
                continue
                
            valid_count += 1
            for cls in base_classes:
                
                # 调用分割方法时传入掩码
                results = self.kb.process_image(
                    image=image,
                    class_name=cls,
                    xml_file=xml_path,
                    save_patches=True
                )
                self.kb.add_to_knowledge(cls, results)
        kb_save_dir = '/root/fsod/model/knowledge_base/kb_pkl/KnowledgeBank.pkl'
        self.kb.save(kb_save_dir)        
        print(f"处理完成：有效图像 {valid_count} 张，缺失标注 {missing_mask_count} 张，知识库已保存到{kb_save_dir}")
        
def build_kb(voc_root, dsr1_key, split_id, print_arch=False):
    builder = KnowledgeBaseBuilder(
        split_id=split_id,
        voc_root=voc_root,
        openai_key=dsr1_key
    )
    knowledge_base = builder.build()
    if print_arch:
        knowledge_base.print_structure()
    return knowledge_base

    
if __name__ == "__main__":
    VOC_ROOT = "/root/dataset/voc/VOCdevkit/VOC2012"
    OPENAI_KEY = "sk-d0347a94786d4bd39287281936ec8ba2"
    SPLIT_ID = 1

    # builder = KnowledgeBaseBuilder(
    #     split_id=SPLIT_ID,
    #     voc_root=VOC_ROOT,
    #     openai_key=OPENAI_KEY
    # )
    
    #knowledge_base = build_kb(VOC_ROOT, OPENAI_KEY, SPLIT_ID)
    kb_path = '/root/fsod/model/knowledge_base/kb_pkl/KnowledgeBank.pkl'
    knowledge_base = Knowledge_Base.load(kb_path, device='cpu')

    knowledge_base.visualize_segmentation(
        image=Image.open("/root/dataset/voc/VOCdevkit/VOC2007/JPEGImages/001420.jpg"),
        save_path="/root/fsod/debug/segmentation_demo.png"
    )
    kb = knowledge_base
    test_img = Image.open("/root/dataset/voc/VOCdevkit/VOC2007/JPEGImages/001420.jpg").convert("RGB")
    patches = kb._segment_image(test_img)  # 取出所有过分割块
    if patches:
        query_embed = kb._encode_images([patches[1]]).squeeze(0)
        query_results = kb.query(query_embed, "horse", top_k=5)
        for attr_name, sim, idx_in_attr in query_results:
            print(attr_name, sim, idx_in_attr)

    if len(patches) > 0:
        for idx in range(len(patches)):
            query_patch = patches[idx]
            kb.query_and_visualize(
                query_patch=query_patch,
                class_name="horse",
                top_k=5,
                save_path=f"/root/fsod/debug/query_result_patch_{idx}.png"  # 或者 None
            )
    kb.print_structure()
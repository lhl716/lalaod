import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.modeling_multimodal_encoder import DummyMultimodalDataset

def train_model(model, 
                dataset, 
                epochs=5, 
                batch_size=2, 
                lr=1e-4,
                device='cuda'):

    # 因为我们的 DummyDataset 是一条一条返回 (没有真正做batch)，
    # 我们可以依靠 DataLoader 的 collate_fn 去做简易的 batch 拼接，
    # 但这里先演示最简方式，batch_size=1 也可以行得通。
    # 如果你需要批量处理，需要在 forward() 做更多处理。
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 仅训练可训练的参数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, (raw_img, text, attr_imgs, label) in enumerate(dataloader):
            # raw_img: [PIL], text: [str], attr_imgs: [list of PIL], label: [int]
            # 由于 batch_size=1，直接取下标 0
            raw_img = raw_img[0]
            text = text[0]
            attr_imgs = attr_imgs[0]
            label = label[0].to(device)

            # 前向计算
            logits = model(raw_img, text, attr_imgs)  # (1, num_classes)
            loss = criterion(logits, label.unsqueeze(0))  # label shape -> (1,)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
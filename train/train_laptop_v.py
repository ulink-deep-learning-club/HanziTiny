#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from model import SimpleViT
import shutil

# ================= 💻 笔记本演示配置 💻 =================
# 强制使用 subset_631 (现有的数据)，但在代码层面只读取前 50 类
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HWDB1.1", "subset_631")

# 强制本地轻量配置
BATCH_SIZE = 128        # 改回 64，这是小数据量收敛的关键
EPOCHS = 180            # 保持 20 轮，甚至可以跑 30
LEARNING_RATE = 1e-3
IMG_SIZE = 64
# ========================================================

# 我们手动定义的前50个高频字 (含"的")
target_chars_str = "一是了我的不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下"
TARGET_CLASSES = list(target_chars_str)
print(f"本次训练仅针对前 {len(TARGET_CLASSES)} 个汉字: {TARGET_CLASSES}")


class LimitedImageFolder(datasets.ImageFolder):
    """
    一个修改版的 ImageFolder，只加载指定的类别文件夹
    """
    def find_classes(self, directory):
        """
        重写：只返回在 TARGET_CLASSES 列表中的文件夹
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        # 过滤：只保留目标字符
        classes = [c for c in classes if c in TARGET_CLASSES]
        
        if not classes:
            raise FileNotFoundError(f"在 {directory} 中没有找到目标类别文件夹！")
            
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# 关键：将 Dataset Wrapper 提到全局，解决 Windows 多进程 Pickle 问题
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def safe_pil_loader(path):
    from PIL import Image
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 
    except Exception:
        return Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据集路径 {DATA_DIR} 不存在！")
        return

    print("正在加载并过滤数据集 (仅 Top 50)...")
    
    # 使用修改后的 ImageFolder
    try:
        full_dataset_raw = LimitedImageFolder(root=DATA_DIR, loader=safe_pil_loader)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"成功加载 {len(full_dataset_raw)} 张图片，覆盖 {len(full_dataset_raw.classes)} 个类别。")

    # 划分数据集
    train_size = int(0.8 * len(full_dataset_raw))
    val_size = len(full_dataset_raw) - train_size
    train_dataset, val_dataset = random_split(full_dataset_raw, [train_size, val_size])

    # 使用全局定义的 Wrapper
    train_dataset = TransformSubset(train_dataset, transform=train_transform)
    val_dataset = TransformSubset(val_dataset, transform=val_transform)

    # 用 num_workers=0 最保险
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型 (注意 num_classes 现在是 50)
    # 把模型搞大点，EMBED_DIM 256
    model = SimpleViT(img_size=IMG_SIZE, patch_size=8, num_classes=len(TARGET_CLASSES), 
                      embed_dim=256, depth=6, num_heads=4, mlp_ratio=4., drop_rate=0.1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # 使用余弦退火学习率，它比 StepLR 平滑多了，非常适合 ViT
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("开始 Demo 训练...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")
            
        scheduler.step()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1} >> Val ACC: {val_acc:.2f}%")

    print("Demo 训练完成！如果看到 Loss 下降且 Acc 上升，说明代码逻辑没问题。")
    
    # 保存模型
    torch.save(model.state_dict(), "laptop_demo_model.pth")
    print("模型已保存至 laptop_demo_model.pth，可用于推理测试。")

if __name__ == '__main__':
    main()

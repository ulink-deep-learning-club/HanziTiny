#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import os
import shutil
from tqdm import tqdm
# from lenet_model import LeNet5
# from modern_net import ModernLeNet
from hanzi_tiny import HanziTiny
import math

# ================= 配置区域 =================
# 数据集路径：使用相对路径，兼容本地和服务器
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HWDB1.1", "subset_631")

def get_config():
    """根据硬件环境动态获取配置"""
    config = {}
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # 这里的 print 稍微有点危险，如果 workers 还是会调这个函数的话。
        # 但我们只会只在 main 里调一次。
        
        if vram_gb > 10: # 大于10G显存 (如 P100 16G) -> 服务器配置
            config['mode'] = "High Performance Mode"
            config['batch_size'] = 512
            config['num_workers'] = 16
            config['embed_dim'] = 768
            config['depth'] = 12
            config['epochs'] = 500
        else: # 小于10G显存 (如 4050 6G) -> 笔记本配置
            config['mode'] = "Local Dev Mode"
            config['batch_size'] = 128
            config['num_workers'] = 4
            config['embed_dim'] = 384 # 稍微增加一点维度提升本地拟合能力
            config['depth'] = 6
            config['epochs'] = 20
    else:
        config['mode'] = "CPU Debug Mode"
        config['batch_size'] = 32
        config['num_workers'] = 0
        config['embed_dim'] = 128
        config['depth'] = 4
        config['epochs'] = 1
        
    config['lr'] = 1e-3
    config['img_size'] = 64
    return config

# ============================================

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

def validate_and_cleanup_data_dir(data_dir):
    """ 清理空文件夹，避免 ImageFolder 报错 """
    if not os.path.exists(data_dir):
        return
    
    removed_count = 0
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp'}
    
    # 遍历所有子目录（类目录）
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_name in subdirs:
        class_path = os.path.join(data_dir, class_name)
        
        # 检查是否有有效图片
        has_valid_file = False
        for f in os.listdir(class_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                has_valid_file = True
                break
        
        # 如果没有有效文件，则移除该目录
        if not has_valid_file:
            print(f"⚠️  警告: 类别 '{class_name}' 为空或无有效图片，正在移除以避免训练报错...")
            # 为了安全起见，这里我们也可以选择不删除，而是重命名，但 ImageFolder 还是会扫描。
            # 用户指示选择移除，所以我们移除。
            try:
                shutil.rmtree(class_path)
                removed_count += 1
            except Exception as e:
                print(f"❌ 移除失败: {e}")
                
    if removed_count > 0:
        print(f"✅ 已自动清理 {removed_count} 个无效类别的文件夹。")

def safe_pil_loader(path):
    """ 安全加载图片，处理异常 """
    from PIL import Image
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') # 转为灰度图
    except Exception as e:
        print(f"警告: 无法读取图片 {path} - {e}")
        # 返回一张全黑图避免崩溃
        return Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)

def main():
    # 1. 获取配置
    config = get_config()
    print(f">> 启用配置: {config['mode']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 数据增强与预处理 (增强版)
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        # 几何变换：旋转、平移、缩放
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5), 
        # 模糊：模拟笔画粗细不均或失焦 (Kernel size 必须是奇数)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # 随机擦除：模拟笔画断裂 (放在 ToTensor 之后)
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)) 
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 3. 加载完整数据集
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据集路径 {DATA_DIR} 不存在！请先运行提取脚本。")
        return

    print("正在预检查数据集完整性...")
    validate_and_cleanup_data_dir(DATA_DIR)

    print("正在加载数据集，请稍候...")
    full_dataset_raw = datasets.ImageFolder(root=DATA_DIR, loader=safe_pil_loader)
    
    num_classes = len(full_dataset_raw.classes)
    print(f"✅ 检测到有效类别数: {num_classes}")

    train_size = int(0.8 * len(full_dataset_raw))
    val_size = len(full_dataset_raw) - train_size
    train_dataset_subset, val_dataset_subset = random_split(full_dataset_raw, [train_size, val_size])
    
    train_dataset = TransformSubset(train_dataset_subset, transform=train_transform)
    val_dataset = TransformSubset(val_dataset_subset, transform=val_transform)

    print(f"数据集划分完成 -> 训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # 5. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=config['num_workers'], pin_memory=True)

    # 6. 初始化模型
    print(f"初始化模型: HanziTiny (针对小样本汉字优化的轻量模型), Classes={num_classes}")
    model = HanziTiny(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05) # 稍微增加 weight_decay
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # 7. 训练循环
    best_acc = 0.0
    EPOCHS = config['epochs']
    
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
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        # 加上进度条，让用户知道没有卡死
        val_loop = tqdm(val_loader, desc=f"Valid [{epoch+1}/{EPOCHS}]", leave=False)
        
        with torch.no_grad():
            for imgs, labels in val_loop:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_loop.set_postfix(loss=loss.item(), acc=f"{100.*val_correct/val_total:.2f}%")
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} 结束: 训练ACC: {train_acc:.2f}%, 验证ACC: {val_acc:.2f}%, 验证Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_modern_lenet.pth")
            torch.save(model.state_dict(), "best_hanzi_tiny.pth")
            
    print("训练全部完成。")

if __name__ == '__main__':
    # 解决 Windows 下多进程 DataLoader 可能报错的问题
    # 同时也确保主入口清晰
    main()

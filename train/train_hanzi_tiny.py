#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import os
import sys
import shutil
from tqdm import tqdm
import math
import argparse # 新增

# 添加项目根目录到 sys.path 以便导入 model
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from model.hanzi_tiny import HanziTiny  # 专用的轻量级汉字识别模型

# ================= 配置区域 =================
# 数据集在根目录下
DATA_DIR = os.path.join(root_dir, "HWDB1.1", "subset_631")

def get_config():
    """根据硬件环境动态获取配置"""
    parser = argparse.ArgumentParser(description='HanziTiny Training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adamw', 'sgd'], help='Optimizer (sgd or adamw)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    args = parser.parse_args()

    config = {}
    
    # HanziTiny 极度轻量，即使在 CPU 上也很快，所以我们可以大胆一点
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb > 8: 
            config['batch_size'] = 512
            config['num_workers'] = 8
            config['epochs'] = 200 # SGD 需要更多的轮次来收敛
        else: 
            config['batch_size'] = 256
            config['num_workers'] = 4
            config['epochs'] = 150
    else:
        config['batch_size'] = 64
        config['num_workers'] = 0
        config['epochs'] = 5
    
    # 如果命令行指定了参数，覆盖默认值
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # 优化器配置
    config['optimizer'] = args.optimizer
    config['patience'] = args.patience
    
    # 学习率调整: SGD 通常需要比 Adam 大得多的 LR
    if args.lr is not None:
        config['lr'] = args.lr
    else:
        # 默认 LR
        if config['optimizer'] == 'sgd':
            config['lr'] = 0.1  # SGD 初始学习率通常较大 (0.1 ~ 0.05)
        else:
            config['lr'] = 2e-3 # AdamW
            
    config['img_size'] = 64
    
    # === 停止条件 ===
    config['target_acc'] = 98.5    # 目标准确率
    
    return config

# ================= 数据集工具 =================

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
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_name in subdirs:
        class_path = os.path.join(data_dir, class_name)
        has_valid_file = False
        for f in os.listdir(class_path):
            if os.path.splitext(f)[1].lower() in valid_exts:
                has_valid_file = True
                break
        
        if not has_valid_file:
            print(f"⚠️  类别 '{class_name}' 为空，移除...")
            try:
                shutil.rmtree(class_path)
                removed_count += 1
            except Exception as e:
                print(f"❌ 移除失败: {e}")
                
    if removed_count > 0:
        print(f"✅ 已清理 {removed_count} 个空类别。")

def safe_pil_loader(path):
    from PIL import Image
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    except Exception as e:
        print(f"无法读取 {path}: {e}")
        return Image.new('L', (64, 64), color=0)

# ================= 主程序 =================

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 HanziTiny 训练 | 设备: {device} | Batch: {config['batch_size']}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)) 
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载数据
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到数据集 {DATA_DIR}")
        return

    validate_and_cleanup_data_dir(DATA_DIR)
    
    print("正在加载数据集索引...")
    full_dataset_raw = datasets.ImageFolder(root=DATA_DIR, loader=safe_pil_loader)
    num_classes = len(full_dataset_raw.classes)
    print(f"✅ 类别数: {num_classes}")

    # === 关键：保存类别映射，确保 GUI 预测时索引一致 ===
    import json
    # 保存到 checkpoints 文件夹
    checkpoints_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 类别映射路径
    class_mapping_path = os.path.join(checkpoints_dir, "classes.json")
    # 状态记录路径 (记录最佳准确率)
    status_path = os.path.join(checkpoints_dir, "train_status.json")
    
    with open(class_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(full_dataset_raw.classes, f, ensure_ascii=False)
    print(f"💾 已保存类别映射到 {class_mapping_path}")

    train_size = int(0.85 * len(full_dataset_raw)) # 小模型不容易过拟合，可以多给点训练集
    val_size = len(full_dataset_raw) - train_size
    train_subset, val_subset = random_split(full_dataset_raw, [train_size, val_size])
    
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=config['num_workers'], pin_memory=True)

    # 初始化模型
    # 模型路径在 checkpoints
    model_path = os.path.join(checkpoints_dir, "best_hanzi_tiny.pth")
    model = HanziTiny(num_classes=num_classes).to(device)

    # === 断点续训逻辑 ===
    best_acc = 0.0
    
    if os.path.exists(model_path):
        print(f"🔄 发现上次训练的最佳模型 {model_path}，准备加载...")
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ 成功加载权重")
            
            # 优先从 status.json 读取上次的准确率，避免因数据集分割不同导致的各种波动
            if os.path.exists(status_path):
                try:
                    with open(status_path, 'r') as f:
                        status = json.load(f)
                        best_acc = status.get('best_acc', 0.0)
                    print(f"📊 从记录文件读取上次最佳准确率: {best_acc:.2f}%")
                except:
                    print("⚠️ 读取 status.json 失败，将重新评估...")
                    best_acc = 0.0
            
            # 如果没有记录或为0，再尝试手动评估 (作为保底)
            if best_acc == 0:
                print("⚠️ 未找到准确率记录，正在重新评估当前验证集基准...")
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
                best_acc = 100. * val_correct / val_total
                print(f"📊 当前模型基准准确率: {best_acc:.2f}%")
            else:
                print(f"📊 继承历史最佳准确率: {best_acc:.2f}%")
            
            # 续训时，建议把学习率调小一点，防止震荡
            # 对于 SGD，如果是续训，可能不需要减半那么激进，或者从一个小一点的值开始
            config['lr'] = config['lr'] * 0.5 
            print(f"📉 续训模式：学习率已自动减半为 {config['lr']}")
            
        except Exception as e:
            print(f"⚠️ 模型加载失败 ({e})，将从头开始训练。")
            best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    no_improve_epochs = 0 # 记录多少轮没提升
    
    # === 优化器选择 ===
    print(f"🔧 使用优化器: {config['optimizer'].upper()} | LR: {config['lr']}")
    if config['optimizer'] == 'sgd':
        # SGD + Momentum 是 CNN 刷分的标配
        # nesterov=True 有时能加速收敛
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    for epoch in range(config['epochs']):
        model.train()
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Ep [{epoch+1}/{config['epochs']}]")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loop.set_postfix(acc=f"{100.*correct/total:.1f}%", loss=f"{loss.item():.3f}")
            
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
        print(f"   -> 验证集准确率: {val_acc:.2f}% (最佳: {best_acc:.2f}%)")

        # 1. 达到目标准确率提前停止
        if val_acc >= config['target_acc']:
            print(f"\n🎯 恭喜！模型已达到目标准确率 {config['target_acc']}%，提前结束训练！")
            if val_acc > best_acc:
                torch.save(model.state_dict(), model_path)
                with open(status_path, 'w') as f:
                    json.dump({'best_acc': val_acc}, f)
            break

        # 2. 保存最佳模型与早停计数
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0 # 重置计数器
            torch.save(model.state_dict(), model_path)
            # 保存状态
            with open(status_path, 'w') as f:
                json.dump({'best_acc': val_acc}, f)
            print(f"   💾 保存最佳模型至 {model_path}")
        else:
            no_improve_epochs += 1
            print(f"   ⏳ 性能未提升 ({no_improve_epochs}/{config['patience']})")
        
        # 3. 触发早停
        if no_improve_epochs >= config['patience']:
            print(f"\n🛑 早停触发：验证集准确率连续 {config['patience']} 轮未提升。")
            print(f"   当前最佳: {best_acc:.2f}%")
            break

    print("\n训练结束。")

if __name__ == '__main__':
    main()

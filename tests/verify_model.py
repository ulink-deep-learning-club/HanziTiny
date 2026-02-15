import torch
import os
import sys
import json
import random
from PIL import Image
from torchvision import transforms

# 添加根目录到 path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from model.hanzi_tiny import HanziTiny

# 配置 (相对于 tests 文件夹)
DATA_DIR = os.path.join(root_dir, "HWDB1.1", "subset_631")
MODEL_PATH = os.path.join(root_dir, "checkpoints", "best_hanzi_tiny.pth")
CLASS_FILE = os.path.join(root_dir, "checkpoints", "classes.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- 核心诊断程序 ---")
    
    # 1. 检查文件
    if not os.path.exists(DATA_DIR):
        print(f"❌ 数据集目录不存在: {DATA_DIR}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    if not os.path.exists(CLASS_FILE):
        print(f"❌ 类别映射不存在: {CLASS_FILE} (请先运行 train_hanzi_tiny.py 生成)")
        return

    # 2. 加载类别
    with open(CLASS_FILE, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    print(f"✅ 加载类别表: {len(classes)} 个类")

    # 3. 加载模型
    try:
        model = HanziTiny(num_classes=len(classes))
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 准备预处理 (完全复刻训练代码)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 5. 随机抽取测试
    test_count = 50
    print(f"\n--- 开始随机抽取 {test_count} 张原图测试 ---")
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # 过滤掉不在 classes 里的文件夹 (防止文件夹变动导致 mismatch)
    valid_subdirs = [d for d in subdirs if d in classes]
    
    correct_count = 0
    
    for i in range(test_count):
        # 随机选一个字
        true_label_name = random.choice(valid_subdirs)
        class_dir = os.path.join(DATA_DIR, true_label_name)
        
        # 随机选这个字的一张图
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        if not images: continue
        
        img_name = random.choice(images)
        img_path = os.path.join(class_dir, img_name)
        
        # 加载推理
        img = Image.open(img_path).convert('L')
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            pred_idx = pred_idx.item()
            pred_label = classes[pred_idx]
            
            is_correct = (pred_label == true_label_name)
            if is_correct: correct_count += 1
            
            status = "✅ 成功" if is_correct else f"❌ 失败 (预测: {pred_label})"
            print(f"样本 {i+1}: 真实=[{true_label_name}] | {status} | 置信度: {conf.item()*100:.1f}%")

    print(f"\n--- 诊断结论 ---")
    if correct_count == test_count:
        print("✅ 模型状态完美！问题一定出在 GUI 的画图处理上。")
    elif correct_count == 0:
        print("❌ 模型彻底乱了！看起来 类别表(classes.json) 和 模型文件(pth) 完全不匹配。")
        print("建议: 删除 classes.json 和 pth 文件，重新运行 train_hanzi_tiny.py。")
    else:
        print("⚠️ 准确率只有部分？模型可能还没训练好，或者过拟合了。")

if __name__ == '__main__':
    main()

import torch
import os
import json

# 假设 train_hanzi_tiny.py 中使用 ImageFolder
# 我们需要确保 GUI 使用的类别映射和训练时完全一致
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HWDB1.1", "subset_631")

def save_class_mapping():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    # 模拟 ImageFolder 的加载逻辑
    # ImageFolder 查找包含有效图片的子文件夹，并按字母顺序排序
    classes = [d.name for d in os.scandir(DATA_DIR) if d.is_dir()]
    classes.sort()
    
    # 过滤掉虽然是文件夹但没有有效图片的类（ImageFolder 会这么做）
    # 但为了简单，先假设所有文件夹都有效，或者训练脚本里并没有再次保存 class_to_idx
    # 最稳妥的方式是在训练脚本里保存这个映射
    
    mapping_path = "class_mapping.json"
    print(f"Saving {len(classes)} classes to {mapping_path}...")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == '__main__':
    save_class_mapping()

#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model import SimpleViT

# ================= 配置 =================
MODEL_PATH = "laptop_demo_model.pth"
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 前50个高频字 (需要与训练时完全一致)
target_chars_str = "一是了我的不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下"
# 关键修复：ImageFolder 使用 sorted() 生成类别索引
TARGET_CLASSES = sorted(list(target_chars_str))
print(f"Using {len(TARGET_CLASSES)} sorted classes.")

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    # 注意参数必须与 train_laptop_v.py 中完全一致
    model = SimpleViT(img_size=IMG_SIZE, 
                      patch_size=8, 
                      num_classes=len(TARGET_CLASSES), 
                      embed_dim=256, 
                      depth=6, 
                      num_heads=4, 
                      mlp_ratio=4., 
                      drop_rate=0.0) # 推理时不 dropout
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # 预处理：转灰度 -> Resize -> Normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Add batch dim
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 获取 Top 3 预测
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            print(f"\n--- Prediction for {os.path.basename(image_path)} ---")
            for i in range(3):
                idx = top3_idx[0][i].item()
                prob = top3_prob[0][i].item()
                char = TARGET_CLASSES[idx]
                print(f"Top {i+1}: 【{char}】 (Probability: {prob*100:.2f}%)")
                
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("请先运行 train_laptop_v.py 生成模型文件！")
    else:
        model = load_model()
        
        # 简单交互式测试
        while True:
            path = input("\n请输入图片路径 (输入 q 退出): ").strip().strip('"') # 去除可能存在的引号
            if path.lower() == 'q':
                break
            predict_image(model, path)

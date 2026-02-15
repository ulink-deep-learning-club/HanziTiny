#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw, ImageTk
import os
import tkinter as tk
from tkinter import messagebox
from lenet_model import LeNet5

# ================= 配置 =================
MODEL_PATH = "best_vit_model.pth"
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取类别列表
# 必须与训练时的顺序完全一致（ImageFolder 默认按文件名排序）
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HWDB1.1", "subset_631")

try:
    if os.path.exists(DATA_DIR):
        TARGET_CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        print(f"✅ 从数据集加载了 {len(TARGET_CLASSES)} 个类别。")
    else:
        # 如果数据集不在了，回退到硬编码列表（不推荐，仅作容错）
        print("⚠️ 警告：未找到数据集文件夹，将使用默认演示列表（可能导致预测错误）")
        TARGET_CHARS_STR = "一是了我的不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下" # 这只是个示例，实际可能不对
        TARGET_CLASSES = sorted(list(TARGET_CHARS_STR))
except Exception as e:
    print(f"无法加载类别列表: {e}")
    TARGET_CLASSES = []

class RobustHandwritingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title(f"汉字识别 (LeNet-5 模型)")
        self.model = model
        
        # 布局容器
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # 左侧：画板
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side='left', padx=10)
        
        self.canvas_size = 280
        # 内存中绘图（白色背景，黑色笔迹）
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image1)
        
        self.canvas = tk.Canvas(left_panel, width=self.canvas_size, height=self.canvas_size, bg='white', cursor="pencil", relief="solid", borderwidth=1)
        self.canvas.pack(pady=5)
        
        # 绑定事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.old_x = None
        self.old_y = None
        
        # 按钮区
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill='x', pady=5)
        
        # 笔画粗细
        controls_frame = tk.Frame(left_panel)
        controls_frame.pack(fill='x', pady=2)
        tk.Label(controls_frame, text="笔画:").pack(side='left')
        self.pen_width = tk.Scale(controls_frame, from_=5, to=40, orient='horizontal', showvalue=0)
        self.pen_width.set(20) 
        self.pen_width.pack(side='left', fill='x', expand=True, padx=5)

        tk.Button(btn_frame, text="识别", command=self.predict, font=("微软雅黑", 12), bg="#e1f5fe", relief="groove").pack(side='left', fill='x', expand=True, padx=2)
        tk.Button(btn_frame, text="清空", command=self.clear, font=("微软雅黑", 12), bg="#ffcdd2", relief="groove").pack(side='right', fill='x', expand=True, padx=2)
        
        # 右侧：结果展示
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side='right', fill='y', padx=10)
        
        # 显示模型看到的输入图
        tk.Label(right_panel, text="模型输入预览:").pack(anchor='w')
        self.debug_label = tk.Label(right_panel, bg='gray', width=128, height=128) # pixel units not working directly for size without image
        self.debug_label.pack(pady=5)
        
        # 识别结果
        tk.Label(right_panel, text="识别结果:", font=("微软雅黑", 10, "bold")).pack(pady=(10,0))
        self.result_char = tk.Label(right_panel, text="?", font=("宋体", 48, "bold"), fg="red")
        self.result_char.pack(pady=5)
        
        self.result_info = tk.Label(right_panel, text="置信度: 0.0%", font=("微软雅黑", 10), fg="#666")
        self.result_info.pack()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def paint(self, event):
        w = self.pen_width.get()
        if self.old_x and self.old_y:
            # 屏幕显示
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=w, fill='black', capstyle=tk.ROUND, smooth=True)
            # 内存绘制
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill='black', width=w, joint='curve')
            
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None
        # 每次抬笔自动预测一次（可选，这里还是手动点按钮稳妥）

    def clear(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.result_char.config(text="?")
        self.result_info.config(text="")
        self.debug_label.config(image='')

    def predict(self):
        if not self.model:
            messagebox.showerror("错误", "模型未加载")
            return
            
        # 1. 获取图片 (白底黑字，和 HWDB 训练集一致)
        # 不要反色，直接使用内存中的图片
        target_img = self.image1
        
        # 2. 预览一下
        debug_view = target_img.resize((128, 128), Image.Resampling.NEAREST)
        self.tk_debug = ImageTk.PhotoImage(debug_view)
        self.debug_label.config(image=self.tk_debug)
        
        # 3. 预处理
        input_tensor = self.transform(target_img).unsqueeze(0).to(DEVICE)
        
        # 4. 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            idx = pred_idx.item()
            probability = conf.item()
            
            if 0 <= idx < len(TARGET_CLASSES):
                char = TARGET_CLASSES[idx]
                self.result_char.config(text=char)
                self.result_info.config(text=f"置信度: {probability*100:.1f}%")
            else:
                self.result_char.config(text="Err")
                self.result_info.config(text="索引越界")

def main():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("错误", f"找不到模型文件: {MODEL_PATH}\n请先运行 train.py 训练模型。")
        return

    num_classes = len(TARGET_CLASSES)
    if num_classes == 0:
        messagebox.showerror("错误", "类别列表为空，无法初始化模型。")
        return

    print(f"Loading LeNet5 model with {num_classes} classes...")
    model = LeNet5(num_classes=num_classes)
    
    try:
        # map_location='cpu' 确保如果没有 GPU 也能加载
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        messagebox.showerror("错误", f"模型加载失败: {e}")
        return

    root = tk.Tk()
    app = RobustHandwritingApp(root, model)
    
    # 居中窗口
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    w = 600
    h = 450
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    root.mainloop()

if __name__ == '__main__':
    main()

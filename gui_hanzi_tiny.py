#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw, ImageTk
import os
import tkinter as tk
from tkinter import messagebox
import sys
# 添加当前目录到 path 以便能找到 model 包（虽然默认就在，但保险）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.hanzi_tiny import HanziTiny # 引入新模型

# ================= 配置 =================
# 模型文件现在位于 checkpoints 文件夹
MODEL_PATH = os.path.join("checkpoints", "best_hanzi_tiny.pth")
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取类别列表
import json
# 数据集目录路径修正 (如果还需要 scan)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HWDB1.1", "subset_631")
# 类别映射文件也在 checkpoints
CLASS_MAPPING_FILE = os.path.join("checkpoints", "classes.json")

def load_classes():
    # 1. 优先尝试从训练脚本生成的 json 加载，这是最准确的
    if os.path.exists(CLASS_MAPPING_FILE):
        try:
            with open(CLASS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                classes = json.load(f)
            print(f"✅ 从 {CLASS_MAPPING_FILE} 加载了 {len(classes)} 个类别 (最可靠)。")
            return classes
        except Exception as e:
            print(f"⚠️ 读取 {CLASS_MAPPING_FILE} 失败: {e}")

    # 2. 如果没有 json，则尝试扫描文件夹 (并剔除空文件夹，模拟 ImageFolder 行为)
    if os.path.exists(DATA_DIR):
        try:
            candidates = sorted([d.name for d in os.scandir(DATA_DIR) if d.is_dir()])
            valid_classes = []
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            
            print("正在扫描有效类别 (可能稍慢)...")
            for cls in candidates:
                cls_path = os.path.join(DATA_DIR, cls)
                # 检查下面有没有图片
                has_img = any(f.name.lower().endswith(tuple(valid_exts)) for f in os.scandir(cls_path) if f.is_file())
                if has_img:
                    valid_classes.append(cls)
            
            print(f"✅ 扫描到 {len(valid_classes)} 个有效类别 (过滤了空文件夹)。")
            return valid_classes
        except Exception as e:
            print(f"扫描失败: {e}")
            return []
            
    return ["Error"]

TARGET_CLASSES = load_classes()

class HanziTinyApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title(f"HanziTiny 极速手写识别 (Acc > 95%)")
        self.model = model
        
        # 布局容器
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # 左侧：画板
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side='left', padx=10)
        
        self.canvas_size = 320 # 稍微大一点
        # 内存中绘图（白色背景，黑色笔迹）- 保持和训练集一致的基础
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image1)
        
        self.canvas = tk.Canvas(left_panel, width=self.canvas_size, height=self.canvas_size, bg='white', cursor="pencil", relief="solid", borderwidth=1)
        self.canvas.pack(pady=5)
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self.start_stroke) # 新增: 按下记录起点
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.old_x = None
        self.old_y = None
        
        # 存储笔画数据 [(x, y), (x, y), ...] 的列表
        self.strokes = [] 
        self.current_stroke = []

        # 按钮区
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill='x', pady=5)
        
        # 笔画粗细
        controls_frame = tk.Frame(left_panel)
        controls_frame.pack(fill='x', pady=2)
        tk.Label(controls_frame, text="笔画:").pack(side='left')
        # command=self.update_width: 拖动滑块时实时重绘
        self.pen_width = tk.Scale(controls_frame, from_=8, to=45, orient='horizontal', showvalue=0, command=self.update_width)
        self.pen_width.set(22) # 针对大一点的画布，笔画也粗一点
        self.pen_width.pack(side='left', fill='x', expand=True, padx=5)

        # 反色按钮 (针对部分模型可能需要黑底白字)
        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls_frame, text="反色 (黑底)", variable=self.invert_var).pack(side='left', padx=5)

        tk.Button(btn_frame, text="识别 (Predict)", command=self.predict, font=("Segoe UI", 12), bg="#e1f5fe", relief="groove").pack(side='left', fill='x', expand=True, padx=2)
        tk.Button(btn_frame, text="清空 (Clear)", command=self.clear, font=("Segoe UI", 12), bg="#ffcdd2", relief="groove").pack(side='right', fill='x', expand=True, padx=2)
        
        # 右侧：结果展示
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side='right', fill='y', padx=20)
        
        # 显示模型看到的输入图
        tk.Label(right_panel, text="Input View:", font=("Segoe UI", 9)).pack(anchor='w')
        self.debug_label = tk.Label(right_panel, bg='gray', width=128, height=128) 
        self.debug_label.pack(pady=5)
        
        # 识别结果
        tk.Label(right_panel, text="Result:", font=("Segoe UI", 10, "bold")).pack(pady=(20,0))
        self.result_char = tk.Label(right_panel, text="?", font=("SimSun", 72, "bold"), fg="red")
        self.result_char.pack(pady=10)
        
        self.result_info = tk.Label(right_panel, text="Conf: 0.0%", font=("Segoe UI", 12), fg="#666")
        self.result_info.pack()

        # Top 3 显示
        tk.Label(right_panel, text="Top 3:", font=("Segoe UI", 9)).pack(pady=(20,5), anchor='w')
        self.top3_label = tk.Label(right_panel, text="", font=("Microsoft YaHei", 10), justify='left')
        self.top3_label.pack(anchor='w')

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def start_stroke(self, event):
        self.old_x = event.x
        self.old_y = event.y
        self.current_stroke = [(event.x, event.y)]

    def paint(self, event):
        w = self.pen_width.get()
        
        # 模拟训练集的灰色字 (灰度值约 70~100)
        # Tkinter 用 hex string
        tk_color = '#404040' 
        # PIL (L model) 用 int (0=黑, 255=白) 
        # 选 60 左右，既不是纯黑也不是太浅
        pil_color = 60 
        
        if self.old_x and self.old_y:
            # 记录轨迹点
            self.current_stroke.append((event.x, event.y))

            # 屏幕显示 (Tkinter 自带圆头优化)
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=w, fill=tk_color, capstyle=tk.ROUND, smooth=True)
            
            # 内存绘制 (Pillow 需要手动画圆来模拟圆头连接，防止断裂)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill=pil_color, width=w)
            
            # 手动补圆：在起点和终点画实心圆，直径等于笔宽
            r = w / 2
            self.draw.ellipse((self.old_x - r, self.old_y - r, self.old_x + r, self.old_y + r), fill=pil_color)
            self.draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=pil_color)
            
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []

    def update_width(self, val):
        self.redraw()

    def redraw(self):
        # 1. 清空画布和图像
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image1)
        
        # 重新获取当前笔宽 (int)
        w = int(self.pen_width.get())
        
        # 颜色定义
        tk_color = '#404040'
        pil_color = 60
        
        # 2. 重绘所有保存的轨迹
        # 注意: self.strokes 里的每一个 stroke 是一个点列表 [(x,y), (x,y), ...]
        for stroke in self.strokes:
            if not stroke: continue
            
            # 单个点（点击）的情况
            if len(stroke) == 1:
                x, y = stroke[0]
                r = w / 2
                x1, y1, x2, y2 = x - r, y - r, x + r, y + r
                self.canvas.create_oval(x1, y1, x2, y2, fill=tk_color, outline=tk_color)
                self.draw.ellipse((x1, y1, x2, y2), fill=pil_color)
                continue

            # Tkinter: 一次性画整条线更平滑
            # flatten list: [(x1,y1), (x2,y2)] -> [x1, y1, x2, y2]
            flat_points = [coord for point in stroke for coord in point]
            
            # capstyle=tk.ROUND 保证线端圆角
            # joinstyle=tk.ROUND 保证拐角圆角
            self.canvas.create_line(flat_points, width=w, fill=tk_color, 
                                  capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=False)
            
            # Pillow: 绘制连贯线条
            self.draw.line(stroke, fill=pil_color, width=w, joint='curve')
            
            # 手动补圆头
            r = w / 2
            # 只需要在每个节点画圆即可覆盖所有连接处
            for p in stroke:
                self.draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=pil_color)
        
        # 3. 如果当前正在画（极其罕见的情况），也重绘一下当前笔画
        if hasattr(self, 'current_stroke') and len(self.current_stroke) > 1:
             w = int(self.pen_width.get())
             stroke = self.current_stroke
             flat_points = [coord for point in stroke for coord in point]
             self.canvas.create_line(flat_points, width=w, fill=tk_color, capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
             self.draw.line(stroke, fill=pil_color, width=w, joint='curve')
             r = w / 2
             for p in stroke:
                self.draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=pil_color)

    def clear(self):
        self.strokes = [] # 清空轨迹记录
        self.current_stroke = []
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.result_char.config(text="?")
        self.result_info.config(text="")
        self.top3_label.config(text="")
        self.debug_label.config(image='')


    def preprocess_image(self, img):
        """
        核心修复：自动裁剪 (Auto-Crop)
        HWDB 数据集里的图片都是切好的字块（字充满画面）。
        但用户在 320x320 的画板上书写时，字可能只占中间一小块，或者偏在一边。
        直接 Resize 会把字缩得非常小，导致模型无法识别。
        """
        # 1. 转为 Numpy 或处理像素寻找边界
        # 这里的 img 是白底黑字 (L mode)
        # 找黑色像素 (值 < 200)，如果字主要是灰色的 (60)，这里 200 没问题
        bbox = img.getbbox() # PIL getbbox 默认找非零像素，但我们是白底(255)，通常用于黑底(0)
        
        # 既然我们是白底黑字，先反色一下再找 bbox 比较方便
        # 反色后：白(255)->黑(0)，灰(60)->亮灰(195)
        # getbbox 会忽略全黑(0)，找到非黑部分
        inverted = ImageOps.invert(img)
        bbox = inverted.getbbox()
        
        if bbox:
            # 找到边界后，稍微往外扩一点 (Padding)，避免字撑得太满贴边
            left, upper, right, lower = bbox
            p = 20 # Padding 像素
            left = max(0, left - p)
            upper = max(0, upper - p)
            right = min(img.width, right + p)
            lower = min(img.height, lower + p)
            
            # 5. 裁剪出来
            img_cropped = img.crop((left, upper, right, lower))
            
            # --- 关键修改：匹配训练时的预处理逻辑 ---
            # 训练代码中使用的是 transforms.Resize((64, 64))，这是不保持长宽比的强制拉伸。
            # 为了让模型“眼熟”我们的输入，我们必须模仿这种“变形”。
            
            return img_cropped

        else:
            # 如果没写字，或者全是白的
            return img

    def predict(self):
        if not self.model:
            messagebox.showerror("错误", "模型未加载")
            return
            
        # 1. 获取图片
        target_img = self.image1
        
        # 2. 如果勾选了反色
        if self.invert_var.get():
            target_img = ImageOps.invert(target_img)

        # 3. 关键步骤：自动裁剪与居中
        target_img = self.preprocess_image(target_img)

        # 3.1 [新增] 模拟扫描件效果：轻微模糊
        # 训练集是扫描件，边缘有灰度过渡。画板画出来的太锐利（纯黑纯白），
        # 导致稍微粗一点就变成了实心块。加一点模糊缓解这个问题。
        from PIL import ImageFilter
        target_img = target_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # 4. 预览一下 (强制用 NEAREST 保持像素感，方便 Debug)
        debug_view = target_img.resize((128, 128), Image.Resampling.NEAREST)
        from PIL import ImageTk
        self.tk_debug = ImageTk.PhotoImage(debug_view)
        # 更新 Label 并显示统计信息
        # 统计一下现在的像素值分布，打印到控制台看看是不是真的变成了黑白分明
        import numpy as np
        arr = np.array(target_img.resize((IMG_SIZE, IMG_SIZE)))
        print(f"[Debug] Img Mean: {arr.mean():.2f}, Min: {arr.min()}, Max: {arr.max()}")
        
        self.debug_label.config(image=self.tk_debug)
        
        # 3.5 预处理 (与训练代码完全一致)
        # 注意：HanziTiny 训练时有 RandomErasing，推理时只有 ToTensor + Normalize
        # ToTensor 会把 [0, 255] 映射到 [0.0, 1.0]
        # Normalize((0.5,), (0.5,)) 会把 [0.0, 1.0] 映射到 [-1.0, 1.0]
        # 白底(255) -> 1.0 -> 1.0
        # 黑字(0) -> 0.0 -> -1.0
        input_tensor = self.transform(target_img).unsqueeze(0).to(DEVICE)
        
        # 4. 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 获取 Top 3
            conf, pred_idx = torch.topk(probs, 3)
            
            top1_idx = pred_idx[0][0].item()
            top1_conf = conf[0][0].item()
            
            # 主结果
            if 0 <= top1_idx < len(TARGET_CLASSES):
                char = TARGET_CLASSES[top1_idx]
                self.result_char.config(text=char)
                self.result_info.config(text=f"Conf: {top1_conf*100:.1f}%")
            
            # Top 3 列表
            top3_text = ""
            for i in range(3):
                idx = pred_idx[0][i].item()
                c = TARGET_CLASSES[idx]
                p = conf[0][i].item()
                top3_text += f"{i+1}. {c}  ({p*100:.1f}%)\n"
            self.top3_label.config(text=top3_text)

def main():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("错误", f"找不到模型: {MODEL_PATH}\n请确认已运行 train_hanzi_tiny.py 并生成了 best_hanzi_tiny.pth")
        return

    num_classes = len(TARGET_CLASSES)
    if num_classes == 0:
        messagebox.showerror("错误", "类别列表为空")
        return

    print(f"Loading HanziTiny model with {num_classes} classes...")
    model = HanziTiny(num_classes=num_classes)
    
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
    app = HanziTinyApp(root, model)
    
    # 居中窗口
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    w = 700
    h = 500
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    root.mainloop()

if __name__ == '__main__':
    main()

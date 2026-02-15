#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import os
import tkinter as tk
from tkinter import messagebox
from model import SimpleViT

# ================= 配置 =================
MODEL_PATH = "laptop_demo_model.pth"
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 前50个高频字 (与训练一致)
TARGET_CHARS_STR = "一是了我的不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下"
# 关键修复：ImageFolder 默认使用 sorted(classes) 来分配索引
# 所以这里必须排序，否则预测结果会乱码
TARGET_CLASSES = sorted(list(TARGET_CHARS_STR))
print(f"Loaded {len(TARGET_CLASSES)} classes (Sorted).")

class HandwritingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("手写汉字识别 Demo (Top 50)")
        self.model = model
        
        # 画布配置
        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white', cursor="pencil")
        self.canvas.pack(pady=10, padx=10)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.old_x = None
        self.old_y = None
        
        # 按钮区域
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="识别", command=self.predict, bg="#DDDDDD", height=2, width=10).pack(side='left', padx=20)
        tk.Button(btn_frame, text="清空", command=self.clear, bg="#DDDDDD", height=2, width=10).pack(side='right', padx=20)
        
        # 结果标签
        self.result_label = tk.Label(root, text="请在写字板上书写...", font=("Microsoft YaHei", 14), fg="blue")
        self.result_label.pack(pady=10)

        # 图像预处理管线
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def paint(self, event):
        if self.old_x and self.old_y:
            # 绘制线条，增加宽度以模拟毛笔/粗笔
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=15, fill='black', capstyle=tk.ROUND, smooth=True)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.canvas.delete("all")
        self.result_label.config(text="请在写字板上书写...")

    def predict(self):
        # 1. 抓取画布内容 (保存为 PostScript 再转 Image)
        # Windows下可能需要 Ghostscript，如果没有安装可能会报错
        # 替代方案：截取画布区域
        try:
            # 这种方法不依赖外部环境，但在高DPI屏幕可能有偏差，简单Demo暂且一用
            # 更稳妥的方法是维护一个隐形的 PIL Image 对象同步绘制，这里为了简单直接截屏画布区域
            
            # 由于 tkinter canvas 转 image 比较麻烦，我们这里创建一个全白的 PIL image
            # 然后把画过的轨迹重新画一遍？太复杂。
            # 简单粗暴：直接生成一张足够大的图
            
            # [兼容性更好的方案]
            # 我们直接在内存里记录了鼠标轨迹吗？没有。
            # 为了确保能跑，我们用一种 trick：
            # 利用 canvas.postscript，如果报错提示安装 Ghostscript，就引导用户
            
            # --- Windows没有Ghostscript怎么搞 ---
            # 我们用 win32gui? 不行太重。
            # 既然是 Demo，我们用 PIL ImageDraw 在内存里同步画一张一模一样的。
            pass
        except Exception:
            pass
            
        # 重新实现：同步绘制
        # 实际上上面的 paint 只是画在屏幕上，我们需要一个 image 对象同步记录
        self.clear() # 这里的逻辑要改，需要在 __init__ 里初始化一个 PIL Image
        messagebox.showerror("错误", "程序内部逻辑需要重置，请重启程序 (开发中)")

# 修正后的实现逻辑
class RobustHandwritingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title(f"汉字识别 (当前模型 Acc 约 17%)")
        self.model = model
        
        self.canvas_size = 280
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        from PIL import ImageDraw
        self.draw = ImageDraw.Draw(self.image1)
        
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white', cursor="pencil")
        self.canvas.pack(pady=10, padx=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.old_x = None
        self.old_y = None
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="识别 (白底黑字)", command=lambda: self.predict(invert=False), font=("微雅软黑", 12), bg="#f0f0f0").pack(side='left', padx=10, ipadx=5)
        tk.Button(btn_frame, text="识别 (黑底白字)", command=lambda: self.predict(invert=True), font=("微雅软黑", 12), bg="#DDDDDD").pack(side='left', padx=10, ipadx=5)
        
        # 笔画粗细控制
        width_frame = tk.Frame(btn_frame)
        width_frame.pack(side='left', padx=20)
        tk.Label(width_frame, text="笔画粗细:").pack(side='left')
        self.pen_width = tk.Scale(width_frame, from_=5, to=30, orient='horizontal')
        self.pen_width.set(18) # 默认值
        self.pen_width.pack(side='left')

        tk.Button(btn_frame, text="清空", command=self.clear, font=("微雅软黑", 12), bg="#f0f0f0").pack(side='right', padx=20, ipadx=10)
        
        self.result_label = tk.Label(root, text="...", font=("Microsoft YaHei", 16), fg="darkblue")
        self.result_label.pack(pady=10)

        # debug 窗口，用于显示 transform 后的图
        self.debug_label = tk.Label(root)
        self.debug_label.pack(pady=5)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def paint(self, event):
        w = self.pen_width.get() # 获取当前笔宽
        if self.old_x and self.old_y:
            # 屏幕显示 (capstyle=ROUND 让线条更圆润，看起来像笔刷)
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=w, fill='black', capstyle=tk.ROUND, smooth=True)
            # 内存图片同步绘制
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill='black', width=w, joint='curve')
            
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        from PIL import ImageDraw
        self.draw = ImageDraw.Draw(self.image1)
        self.result_label.config(text="...")

    def predict(self, invert=False):
        # 准备图片
        target_img = self.image1
        
        # 如果需要反色 (变成黑底白字)
        if invert:
            target_img = ImageOps.invert(target_img)
            
        # 调试：显示送入模型的图片长什么样
        # Resize 到 64x64 方便观察
        debug_img = target_img.resize((64, 64))
        # 放大一点显示在界面上
        display_debug = debug_img.resize((128, 128), Image.Resampling.NEAREST)
        from PIL import ImageTk
        self.tk_debug = ImageTk.PhotoImage(display_debug)
        self.debug_label.config(image=self.tk_debug, text=f"Input ({'Inverted' if invert else 'Original'})", compound='bottom')
        
        input_tensor = self.transform(target_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            idx = pred_idx.item()
            char = TARGET_CLASSES[idx]
            probability = conf.item()
            
            # 显示 Top 3
            top3_prob, top3_idx = torch.topk(probs, 3)
            res_str = f"Top1: 【{char}】 ({probability*100:.1f}%)\n"
            res_str += f"Top2: {TARGET_CLASSES[top3_idx[0][1]]} ({top3_prob[0][1]*100:.1f}%)"
            
            self.result_label.config(text=res_str)
            print(f"Pred: {char}, Conf: {probability:.4f} (Invert={invert})")

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleViT(img_size=IMG_SIZE, 
                      patch_size=8, 
                      num_classes=len(TARGET_CLASSES), 
                      embed_dim=256,   # 你的新参数
                      depth=6,         # 你的新参数
                      num_heads=4, 
                      mlp_ratio=4., 
                      drop_rate=0.0)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError:
        print("模型参数不匹配！请确保 MODEL_PATH 指向的是最新 EMBED_DIM=256 的模型。")
        return None
        
    model.to(DEVICE)
    model.eval()
    return model

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("找不到模型文件。")
    else:
        model = load_model()
        if model:
            root = tk.Tk()
            app = RobustHandwritingApp(root, model)
            # 居中窗口
            root.eval('tk::PlaceWindow . center')
            root.mainloop()

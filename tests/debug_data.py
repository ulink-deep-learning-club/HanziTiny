from PIL import Image
import os
import torchvision.transforms as T

def check_raw_image():
    test_folder = os.path.join("HWDB1.1", "subset_631", "阿")
    # 直接硬编码一个文件名，因为 ls 结果太长了
    # 我刚才看到了 dir 列表，第一个通常是 0.png 或类似
    # 我们遍历找第一个
    if not os.path.exists(test_folder):
        return

    imgs = [f for f in os.listdir(test_folder) if f.endswith(".png") or f.endswith(".jpg")]
    if not imgs:
        return
        
    img_path = os.path.join(test_folder, imgs[0])
    print(f"\n>>>> 正在分析训练数据样本: {img_path}")
    
    img = Image.open(img_path)
    img = img.convert('L') # 必须转灰度
    
    # 打印 corners (背景) 和 center (前景，假设字在中间)
    w, h = img.size
    bg = img.getpixel((0, 0)) # 左上角
    fg = img.getpixel((w//2, h//2)) # 中心点
    
    print(f"图像尺寸: {w}x{h}")
    print(f"背景像素值 (Top-Left): {bg}  ({ '白色/亮色' if bg > 128 else '黑色/暗色' })")
    print(f"中心像素值 (Center):   {fg}")
    
    # 模拟 Train Loader 里的行为
    import torchvision.transforms as T
    trans = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    tensor = trans(img)
    print(f"\nTensor 统计: Min={tensor.min():.2f}, Max={tensor.max():.2f}, Mean={tensor.mean():.2f}")
    
    if tensor[0, 0, 0] > 0:
        print(">> 结论: 模型训练时输入的是【白底黑字】(背景接近 1.0)")
        print(">> ！！！GUI 必须输入白底黑字，且不能反色！！！")
    else:
        print(">> 结论: 模型训练时输入的是【黑底白字】(背景接近 -1.0)")
        print(">> ！！！GUI 必须输入黑底白字 (通常写字板是白底，所以必须反色)！！！")

if __name__ == '__main__':
    check_raw_image()

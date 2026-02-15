import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    SE (Squeeze-and-Excitation) 注意力模块
    作用：让模型自动学习每一个特征通道的权重。
    比如：模型发现当前是“人”字，它会自动调高“撇”和“捺”对应特征通道的权重。
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DSConv(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Convex)
    这是 MobileNet 的核心。
    
    1. Depthwise (dw): 3x3 卷积，groups=in_chans。
       它独立地对每个通道进行空间卷积。因为是 3x3，所以它能完美捕捉 ↘ ↙ 这样的斜向笔画。
       参数量: 3 * 3 * C = 9C
       对比 ModernLeNet 的 5x1+1x5: 5C + 5C = 10C
       结论：我们用更少的参数，换来了全向（包括斜向）的感知能力。
       
    2. Pointwise (pw): 1x1 卷积。
       负责把 dw 提取的特征融合起来。
       
    3. SEBlock: 加上注意力机制，进一步增强关键特征。
    """
    def __init__(self, in_chans, out_chans, stride=1, use_se=True):
        super().__init__()
        
        # 1. 深度卷积 (提取形状、笔画)
        self.dw = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=stride, padding=1, groups=in_chans, bias=False),
            nn.BatchNorm2d(in_chans),
            nn.ReLU6(inplace=True) # ReLU6 对低精度计算更友好
        )
        
        # 2. 注意力 (可选)
        self.se = SEBlock(in_chans) if use_se else nn.Identity()
        
        # 3. 逐点卷积 (融合特征)
        self.pw = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.se(x)
        x = self.pw(x)
        return x

class HanziTiny(nn.Module):
    """
    专为极小数据集汉字识别设计的超轻量模型。
    参数量极低，防止过拟合；3x3 DW卷积确保斜向特征提取。
    """
    def __init__(self, num_classes=630, in_chans=1):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Stem层: 快速降低分辨率，提取基础特征
        # 64x64 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # 骨干网络 (Backbone)
        self.features = nn.Sequential(
            # 32x32
            # 这里的 hidden channel 稍微给大一点，保证特征丰富度
            DSConv(32, 64, stride=1),
            nn.MaxPool2d(2), # 32 -> 16
            
            # 16x16
            DSConv(64, 128, stride=1),
            DSConv(128, 128, stride=1),
            nn.MaxPool2d(2), # 16 -> 8
            
            # 8x8
            DSConv(128, 256, stride=1),
            # 最后只需要 Global Average Pooling，不需要 Flatten 后接巨大的 FC 层
        )
        
        # 简单的分类头
        # 相比 LeNet 这里的 FC 层参数量非常非常小，因为输入只有 256 个数
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        # x: [B, 1, 64, 64]
        x = self.stem(x) 
        x = self.features(x)
        
        # Global Average Pooling: [B, 256, 8, 8] -> [B, 256, 1, 1]
        # 把每个通道的 8x8 特征图平均成一个数，这样位置信息就无关了，只看有没有这个特征
        x = x.mean([2, 3]) 
        
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

if __name__ == '__main__':
    model = HanziTiny(num_classes=630)
    print("模型结构创建成功")
    
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HanziTiny 总参数量: {params/1000:.1f}k (LeNet约400-500k, ViT约数M)")
    
    dummy = torch.randn(2, 1, 64, 64)
    out = model(dummy)
    print("输出形状:", out.shape)

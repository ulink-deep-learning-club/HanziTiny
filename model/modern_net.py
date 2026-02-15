import torch
import torch.nn as nn

class Conv(nn.Module):
    """ 封装好的卷积块：Conv2d + BatchNorm + Activation """
    def __init__(self, ch_in: int, ch_out: int, kernel_size: tuple = (3, 3), 
                 act: nn.Module = None, bn: bool = True):
        super().__init__()
        # 计算 padding 保证尺寸逻辑（这里简单处理，尽量保持 valid 或 same）
        # 原代码没写 padding，但在 MyNet 逻辑里似乎是利用 valid padding 做降维
        # 这里严格复刻原 MyNet 的逻辑（不加 padding）
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(ch_out) if bn else nn.Identity()
        self.act = act if act else nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Linear(nn.Module):
    """ 封装好的全连接块：Linear + Activation + Dropout """
    def __init__(self, feat_in: int, feat_out: int, bias: bool = True, 
                 act: nn.Module = None, dropout: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(feat_in, feat_out, bias)
        self.act = act if act else nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.linear(x)))

class ModernLeNet(nn.Module):
    """
    针对汉字优化的改进版 LeNet
    特点：
    1. 使用 5x1 和 1x5 非对称卷积提取笔画特征
    2. 引入 BatchNorm 加速收敛
    3. 引入 SiLU 激活函数
    4. 引入 Dropout 防止小数据过拟合
    """
    
    def __init__(self, num_classes: int = 630, input_channels: int = 1, input_size: int = 64):
        super().__init__()
        
        self.channels = 16
        
        # 特征提取
        self.features = nn.Sequential(
            # Layer 1: 常规 5x5 卷积
            Conv(input_channels, self.channels, (5, 5)), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: 非对称卷积分解
            # 5x1 卷积 (提取竖向特征)
            Conv(self.channels, self.channels, (5, 1)),
            # 1x5 卷积 (提取横向特征)
            Conv(self.channels, self.channels, (1, 5)),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 动态计算 Flatten 后的大小
        # 依照 64x64 输入手动推导：
        # 1. Conv 5x5 -> 60x60 -> Pool 2x2 -> 30x30
        # 2. Conv 5x1 -> 26x30
        # 3. Conv 1x5 -> 26x26 -> Pool 2x2 -> 13x13
        # 最终: 16通道 * 13 * 13
        
        # 为了兼容性，使用公式计算
        h, w = (input_size, input_size)
        
        # Layer 1 block
        h = (h - 5) // 2 
        w = (w - 5) // 2
        
        # Layer 2 block
        # Conv 5x1
        h = h - 5 + 1
        w = w - 1 + 1
        # Conv 1x5
        h = h - 1 + 1
        w = w - 5 + 1
        # Pool
        h = h // 2
        w = w // 2
        
        self.feature_size = self.channels * h * w
        
        self.classifier = nn.Sequential(
            Linear(self.feature_size, 128, dropout=0.2), # 稍微加大一点中间层
            Linear(128, 64, dropout=0.3),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

if __name__ == '__main__':
    model = ModernLeNet(num_classes=630, input_size=64)
    print("Feature Size:", model.feature_size)
    dummy = torch.randn(2, 1, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)

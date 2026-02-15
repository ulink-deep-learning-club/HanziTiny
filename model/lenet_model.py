import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 模型 (适配 64x64 输入)
    
    经典结构:
    1. Conv1: 1 -> 6 (5x5)
    2. MaxPool: 2x2
    3. Conv2: 6 -> 16 (5x5)
    4. MaxPool: 2x2
    5. FC1: Flatten -> 120
    6. FC2: 120 -> 84
    7. FC3: 84 -> num_classes
    """
    def __init__(self, num_classes=630, in_chans=1):
        super(LeNet5, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_chans, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 对于 64x64 输入:
        # Conv1: (64 - 5 + 1) = 60 -> Pool: 30
        # Conv2: (30 - 5 + 1) = 26 -> Pool: 13
        # 输出尺寸: 16 * 13 * 13
        self.feature_size = 16 * 13 * 13
        
        # 分类头部分
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 测试代码
    model = LeNet5(num_classes=630)
    print("模型结构:")
    print(model)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")
    
    # 测试输入
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

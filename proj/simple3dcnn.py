from torch import nn
import torch.nn.functional as F
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        # 输入维度为(1, 20, 192, 192)，假设数据为单通道
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1) # 输出维度: (16, 20, 192, 192)
        self.pool = nn.MaxPool3d(2, 2) # 输出维度: (16, 10, 96, 96)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1) # 输出维度: (32, 10, 96, 96)
        # 可以添加更多的卷积层和池化层...
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1) # 输出维度: (1, 20, 192, 192)
        self.conv5 = nn.Conv3d(128, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
        self.conv6 = nn.Conv3d(64, 32, kernel_size=3, padding=1)# 输出维度: (1, 20, 192, 192)
        self.conv7 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(16, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.up(x)
        x = self.conv8(x)
        return x

# 模型实例化

# 假设你有一个符合形状要求的输入
# input_tensor = torch.randn(1, 1, 20, 192, 192) # (batch_size, channels, depth, height, width)
# output = model(input_tensor)
# print(output.shape)

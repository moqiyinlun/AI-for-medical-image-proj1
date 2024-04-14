from torch import nn
from torch.nn import functional as F
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1) # 输出维度: (16, 20, 192, 192)
        self.pool = nn.MaxPool3d(2, 2) # 输出维度: (16, 10, 96, 96)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1) # 输出维度: (32, 10, 96, 96)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1) # 输出维度: (1, 20, 192, 192)
        self.conv5 = nn.Conv3d(128, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
        self.conv6 = nn.Conv3d(64, 32, kernel_size=3, padding=1)# 输出维度: (1, 20, 192, 192)
        self.conv7 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(16, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = self.up(out)
        out = self.conv8(out)
        out += identity
        return out

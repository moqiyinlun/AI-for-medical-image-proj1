from torch import nn
from torch.nn import functional as F
#使用resent的3dCNN
#100 epoch 29 90
# class ResBlock(nn.Module):
#     def __init__(self):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1) # 输出维度: (16, 20, 192, 192)
#         self.pool = nn.MaxPool3d(2, 2) # 输出维度: (16, 10, 96, 96)
#         self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1) # 输出维度: (32, 10, 96, 96)
#         # 可以添加更多的卷积层和池化层...
#         #dropout

#         self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
#         self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1) # 输出维度: (1, 20, 192, 192)
#         self.conv5 = nn.Conv3d(128, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
#         self.conv6 = nn.Conv3d(64, 32, kernel_size=3, padding=1)# 输出维度: (1, 20, 192, 192)
#         self.conv7 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
#         self.conv8 = nn.Conv3d(16, 1, kernel_size=3, padding=1)
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.conv1(x))
#         out = self.pool(out)
#         out = F.relu(self.conv2(out))
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         out = F.relu(self.conv5(out))
#         out = F.relu(self.conv6(out))
#         out = F.relu(self.conv7(out))
#         out = self.up(out)
#         out = self.conv8(out)
#         out += identity
#         return out


#100 epoch 29 90
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1) # 输出维度: (16, 20, 192, 192)
        self.pool1 = nn.MaxPool3d(2, 2) # 输出维度: (16, 10, 96, 96)
        self.pool2 = nn.MaxPool3d(2, 2) # 输出维度: (16, 10, 96, 96)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1) # 输出维度: (32, 10, 96, 96)
        # 可以添加更多的卷积层和池化层...
        #dropout

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1) # 输出维度: (16, 10, 96, 96)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1) # 输出维度: (1, 20, 192, 192)
        self.conv5 = nn.Conv3d(128, 64, kernel_size=3 , padding=1)
        self.conv6 = nn.Conv3d(64, 32, kernel_size=3, padding=1)# 输出维度: (1, 20, 192, 192)
        self.conv7 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(16, 1, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = self.up1(out)
        out = self.conv8(out)
        out += identity
        return out

class ResBlock_time(nn.Module):
    def __init__(self):
        super(ResBlock_time, self).__init__()
        # 空间卷积层
        self.conv2d1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.pool2d1 = nn.MaxPool2d(2, 2)  # 空间下采样
        self.conv2d2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))

        # 时间卷积层，处理时间维度，用于提取时间信息
        self.conv1d1 = nn.Conv1d(20, 20, kernel_size=3, padding=1, groups=20)  # 时间维度卷积

        # 后续2D卷积层
        self.conv2d3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d5 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d6 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d7 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1))
        self.up2d1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 空间上采样

    def forward(self, x):
        batch_size, channels, time, height, width = x.size()

        # 先处理所有空间信息
        x = x.view(batch_size * time, channels, height, width)  # 将时间维和批次维合并，对每个时间帧单独处理空间信息
        x = F.relu(self.conv2d1(x))
        x = self.pool2d1(x)
        x = F.relu(self.conv2d2(x))
        x = x.view(batch_size, time, 32, height // 2, width // 2)  # 恢复时间维度

        # 通过1D卷积处理时间信息
        x_time = x.permute(0, 2, 1, 3, 4).contiguous()  # 调整维度以将时间维度放到卷积位置
        x_time = x_time.view(batch_size * 32, time, height // 2 * width // 2)  # 合并空间维度进行1D卷积
        x_time = F.relu(self.conv1d1(x_time))
        x_time = x_time.view(batch_size, 32, time, height // 2, width // 2)  # 恢复维度
        x_time = x_time.permute(0, 2, 1, 3, 4).contiguous()  # 恢复原始维度顺序

        # 继续后续的2D卷积操作
        x = x_time.view(batch_size * time, 32, height // 2, width // 2)
        x = F.relu(self.conv2d3(x))
        x = F.relu(self.conv2d4(x))
        x = F.relu(self.conv2d5(x))
        x = F.relu(self.conv2d6(x))
        x = self.up2d1(x)  # 上采样回原始高宽
        x = F.relu(self.conv2d7(x))
        x = x.view(batch_size,1, time, height, width)  # 重新调整维度以匹配输入格式

        return x
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
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(5,3,3), padding=1) # 输出维度: (16, 20, 192, 192)
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
        self.conv2d1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.conv2d2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.conv1d = nn.Conv1d(32 * 48 * 48, 32 * 48 * 48, kernel_size=3, padding=1, groups=32 * 48 * 48)

        self.conv2d3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2d4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2d5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2d6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv2d7 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv2d8 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')  

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        identity = x.clone()

        out = x.view(batch_size * depth, channels, height, width)
        out = F.relu(self.conv2d1(out))
        out = self.pool1(out)
        out = F.relu(self.conv2d2(out))
        out = out.view(batch_size, depth, 32, height // 2, width // 2)
        out = out.permute(0, 2, 3, 4, 1).contiguous() 
        out = out.view(batch_size, 32 * (height // 2) * (width // 2), depth)
        out = F.relu(self.conv1d(out))
        out = out.view(batch_size, 32, height // 2, width // 2, depth).permute(0, 4, 1, 2, 3)

        out = out.view(batch_size * depth, 32, height // 2, width // 2)
        out = F.relu(self.conv2d3(out))
        out = F.relu(self.conv2d4(out))
        out = F.relu(self.conv2d5(out))
        out = F.relu(self.conv2d6(out))
        out = F.relu(self.conv2d7(out))
        out = self.up1(out)
        out = self.conv2d8(out)
        out = out.view(batch_size,  1,depth ,height, width)  

        out += identity
        return out
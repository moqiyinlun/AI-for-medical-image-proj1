from proj.resblock import ResBlock
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torch import nn
import torch
import torch.nn.functional as F
from PIL import Image
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        total_perceptual_loss = 0.0
        batch_size, channels, depth, height, width = output.size()
        repeat_factor = [1, 3, 1, 1, 1]  
        output = output.repeat(repeat_factor)  
        target = target.repeat(repeat_factor)  
        for i in range(batch_size):
            for d in range(depth):
                output_slice = output[i, :, d, :, :]
                target_slice = target[i, :, d, :, :]
                output_features = self.vgg(output_slice)
                target_features = self.vgg(target_slice)
                total_perceptual_loss += F.mse_loss(output_features, target_features, reduction='sum')
        return total_perceptual_loss
def edge_loss(y_pred, y_true):
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=y_pred.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=y_pred.device)
    grad_pred_x = torch.zeros_like(y_pred)
    grad_pred_y = torch.zeros_like(y_pred)
    grad_true_x = torch.zeros_like(y_true)
    grad_true_y = torch.zeros_like(y_true)
    for b in range(y_pred.shape[0]):
        for t in range(y_pred.shape[2]):
            grad_pred_x[b, :, t] = F.conv2d(y_pred[b, :, t:t+1], sobel_x, padding=1)
            grad_pred_y[b, :, t] = F.conv2d(y_pred[b, :, t:t+1], sobel_y, padding=1)
            plt.imshow(y_pred[b,0,t].cpu().detach().numpy(),cmap='gray')
            plt.show()
            grad_true_x[b, :, t] = F.conv2d(y_true[b, :, t:t+1], sobel_x, padding=1)
            grad_true_y[b, :, t] = F.conv2d(y_true[b, :, t:t+1], sobel_y, padding=1)
    grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2)
    grad_true = torch.sqrt(grad_true_x ** 2 + grad_true_y ** 2)
    return F.l1_loss(grad_pred, grad_true)
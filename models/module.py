import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
#              SAFE              #
##################################

def add_noise_safe(tensor, noise_std_ratio=0.6):
    std = torch.std(tensor)
    noise_std = std * noise_std_ratio
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise

class SAFEModule(nn.Module):
    def __init__(self, in_channels, out_channels=64, noise_std_ratio=0.6):
        super(SAFEModule, self).__init__()
        self.noise_std_ratio = noise_std_ratio

        # Depthwise Separable Conv
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Multi-scale dilated convs
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)

        # Channel attention (Squeeze-Excitation)
        self.se_fc1 = nn.Linear(out_channels * 3, out_channels)
        self.se_fc2 = nn.Linear(out_channels, out_channels * 3)

        # Skip connection
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.elu = nn.ELU()
        self.celu = nn.CELU()
        self.gelu = nn.GELU()

    def forward(self, x):
        # Depthwise + Pointwise
        out1 = self.depthwise(x)
        out1 = self.pointwise(out1)
        out1 = self.gelu(out1)

        out2 = self.depthwise(x)
        out2 = self.pointwise(out2)
        out2 = self.elu(out2)

        out3 = self.depthwise(x)
        out3 = self.pointwise(out3)
        out3 = self.relu(out3)
        
        out = out1*out2*out3

        # Multi-scale dilated convs
        dil1 = self.gelu(self.dil_conv1(out))
        dil2 = self.gelu(self.dil_conv2(out))
        dil4 = self.gelu(self.dil_conv4(out))
        multi_scale1 = torch.cat([dil1, dil2, dil4], dim=1)

        # Multi-scale dilated convs
        dil1 = self.elu(self.dil_conv1(out))
        dil2 = self.elu(self.dil_conv2(out))
        dil4 = self.elu(self.dil_conv4(out))
        multi_scale2 = torch.cat([dil1, dil2, dil4], dim=1)

        # Multi-scale dilated convs
        dil1 = self.relu(self.dil_conv1(out))
        dil2 = self.relu(self.dil_conv2(out))
        dil4 = self.relu(self.dil_conv4(out))
        multi_scale3 = torch.cat([dil1, dil2, dil4], dim=1)

        multi_scale= multi_scale1*multi_scale2*multi_scale3

        # Channel attention (SE block)
        se = F.adaptive_avg_pool2d(multi_scale, 1).view(multi_scale.size(0), -1)
        se = self.relu(self.se_fc1(se))
        se = self.relu(self.se_fc2(se))
        se = se.view(multi_scale.size(0), multi_scale.size(1), 1, 1)
        attended = multi_scale * se

        # Fuse + Skip
        fused = self.elu(self.fuse(attended))
        skip = self.skip(x)

        return fused + skip

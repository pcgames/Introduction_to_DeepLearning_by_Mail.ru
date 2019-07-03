import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
       )

    def forward(self, x):
        x = self.conv(x)
        return x

#Downsample
class DSBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DSBlock, self).__init__()
        self.mp = nn.Sequential(
            nn.MaxPool2d(2, 2),
            double_conv(in_c, out_c)
        )

    def forward(self, x):
        x = self.mp(x)
        return x
#Upsample
class USBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(USBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_c*2, out_c, 3, padding=1)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        x1=torch.cat([x1,x2],dim=1)
        x = self.conv(x1)
        return x



class SegmenterModel(nn.Module):
    def __init__(self):
        super(SegmenterModel, self).__init__()
        self.conv0 = double_conv(3, 64)
        self.down1 = DSBlock(64, 128)
        self.down2 = DSBlock(128, 256)
        self.down3 = DSBlock(256, 512)
        self.conv1 = double_conv(512, 512)
        self.down4 = DSBlock(512, 512)
        self.conv2 = double_conv(512, 512)
        
        self.up1 = USBlock(512, 256)
        self.conv3 = double_conv(256, 256)
        self.up2 = USBlock(256, 128)
        self.up3 = USBlock(128, 64)
        self.up4 = USBlock(64, 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.conv1(x3)
        x5 = self.down4(x4)
        x5 = self.conv2(x5)
        
        x3 = self.up1(x5,x3)
        x3 = self.conv3(x3)

        x2 = self.up2(x3,x2)
        x1 = self.up3(x2,x1)
        x = self.up4(x1,x)
        x = self.final_conv(x)
        return x
    
    def predict(self, x):
        out = self.forward(x.unsqueeze(0).cuda())
        out = out > 0
        out = out.squeeze(0).squeeze(0).float().cuda()
        return out

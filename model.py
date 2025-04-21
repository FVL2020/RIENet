import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

class L_rec(nn.Module):
    def __init__(self, lambda_=0.8, eps=1e-3):
        super(L_rec, self).__init__()
        self.lambda_ = lambda_
        self.eps = eps
        self.ssim = SSIM()
    def forward(self, x, x_hat):
        loss = (1-self.lambda_)*torch.sqrt(torch.pow(x-x_hat, 2).mean()+self.eps**2) - self.lambda_*self.ssim(x, x_hat)
        return loss
    
class RIENet(nn.Module):
    def __init__(self, config):
        super(RIENet, self).__init__()
        self.srnet = ENS()
        self.ignet = IRS()
        self.csnet = RIEM(self.ignet.len_z, config.warm_up)
        self.w_h = config.w_h
        self.w_z = config.w_z
        self.l_rec = L_rec(config.lambda_)
        self.config = config
    def forward(self, x, ref):
        h = self.srnet(x)
        z_ref = self.csnet(ref)
        x_enhance = self.ignet(h,z_ref)
        return x_enhance, h, z_ref


    def curve_synthesis(self, x):
        with torch.no_grad():
            alpha = (torch.rand(1)-0.5)*2
            alpha = alpha.cuda()
            iteration_n = np.random.randint(1,5)
            for i in range(iteration_n):
                x = x + alpha*x*(1-x)
        return x.detach()
    def loss(self, x, gt):
        l_rec = self.l_rec
        x_enhance, h, z_gt = self.forward(x, gt)
        h_gt, n_xgt = self.srnet(gt)
        if not self.w_h == 0:
            loss_h = self.w_h*F.mse_loss(h, h_gt)
        else:
            loss_h = torch.tensor([0]).to(dtype=x.dtype,device=x.device)
        loss_rec =  l_rec(x_enhance, gt)

        # different brightness correspond to different z, correspond to different output
        if not self.w_z == 0:
            new_h = h.detach()
            if not self.config.dataset == 'EE':
                gt_adjust = self.curve_synthesis(gt)
                z_gt_adjust = self.csnet(gt_adjust)
                loss_z = self.w_z*l_rec(self.ignet(new_h,z_gt_adjust),gt_adjust)
            else:
                z_x = self.csnet(x)
                loss_z = self.w_z*l_rec(self.ignet(new_h,z_x),x)
        else:
            loss_z = torch.tensor([0]).to(dtype=x.dtype,device=x.device)

        loss = loss_h + loss_z + loss_rec
        return loss, loss_h, loss_rec, loss_z



class ENS(nn.Module):
    def __init__(self):
        super(ENS, self).__init__()
        self.in_channels = [3,32,32,32,64,64,64]
        self.out_channels = [32,32,32,32,32,32,32]
        self.input_norm = nn.InstanceNorm2d(num_features=3)
        self.unet = torch.nn.ModuleList([
            nn.Sequential(
                nn.InstanceNorm2d(num_features=self.in_channels[i]),
                nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=3, stride=1, padding='same', padding_mode='reflect'),
                nn.LeakyReLU()) for i in range(7)
        ])
        self.unet_res = torch.nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.out_channels[i], out_channels=self.out_channels[i], kernel_size=3, stride=1, padding='same', padding_mode='reflect'),
                nn.LeakyReLU()) for i in range(7)
        ])
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels[-1], out_channels=3, kernel_size=1, stride=1),
        )
    def forward(self, x):
        feature_list = []
        for i in range(3):
            x = self.unet[i](x)
            x = self.unet_res[i](x)*0.2+x
            feature_list.append(x)
            x = F.interpolate(x,scale_factor=1/2,mode='bilinear')
        for i in range(3):
            x = self.unet[i+3](x)
            x = self.unet_res[i+3](x)*0.2+x
            x = F.interpolate(x,scale_factor=2,mode='bilinear')
            x = torch.cat([x, feature_list[-1-i]], dim=1)
        x = self.unet[6](x)
        x = self.unet_res[6](x)*0.2+x
        out = self.out_conv(x)
        return out

class IRS(nn.Module):
    def __init__(self):
        super(IRS, self).__init__()
        self.in_channels = [3,48,48,48,48]
        self.conv_net = torch.nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.in_channels[i+1], kernel_size=3, stride=1, padding='same', groups=3),
                nn.LeakyReLU()) for i in range(len(self.in_channels)-1)
        ])
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels[-1], out_channels=3, kernel_size=1, stride=1, groups=3),
        )
        self.start_index = [0]
        for i in range(1, len(self.in_channels)-1):
            len_previous = 2*self.in_channels[i]
            self.start_index.append(self.start_index[i-1]+len_previous)
        self.len_z = self.start_index[-1]+2*self.in_channels[-1]
    def forward(self, x, z):
        for i in range(len(self.in_channels)-1):
            if z is not None:
                start_index = self.start_index[i]
                end_index1 = start_index + self.in_channels[i+1]
                end_index2 = end_index1 + self.in_channels[i+1]
                scale = z[:,start_index:end_index1].unsqueeze(dim=2).unsqueeze(dim=2)+1
                bias = z[:,end_index1:end_index2].unsqueeze(dim=2).unsqueeze(dim=2)
                x = self.conv_net[i](x)*scale+bias
            else:
                x = self.conv_net[i](x)
        out = self.out_conv(x)
        return out

class RIEM(nn.Module):
    def __init__(self, output_len, warm_up = 1):
        super(RIEM, self).__init__()
        self.warm_up = warm_up
        self.in_channels = [3,16,64,256]
        self.fc_channels = [256,256,256]
        self.conv_net = torch.nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.in_channels[i+1], kernel_size=3, stride=1, padding='same'),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=2)) for i in range(len(self.in_channels)-1)
        ])
        self.fc_net = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=self.fc_channels[i], out_features=self.fc_channels[i+1]),
                nn.LeakyReLU()) for i in range(len(self.fc_channels)-1)
        ])
        self.out_fc = nn.Linear(in_features=self.fc_channels[-1], out_features=output_len)
    def forward(self, x):
        for conv_i in self.conv_net:
            x = conv_i(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(dim=2).squeeze(dim=2)
        for fc_i in self.fc_net:
            x = fc_i(x)
        out = self.out_fc(x)
        out = F.tanh(out)
        return out*self.warm_up

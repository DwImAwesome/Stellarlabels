import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import random

from numpy import genfromtxt

import numpy as np
import cv2
import torchvision.utils as utils
from utils import params_to_tb, add_noise

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=(1,15), padding=(0,7), bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.AvgPool2d(kernel_size=(1,3), stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g
        
class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
        return c.view(N,1,W,H), output


class Attention_network(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(Attention_network, self).__init__()
        self.attention = attention
        # Convolutional blocks
        self.conv_block1 = ConvBlock(1, 64, 1)
        self.conv_block3 = ConvBlock(64, 128, 1)
        self.conv_block4 = ConvBlock(128, 128, 3)
        self.conv_block5 = ConvBlock(128, 128, 3)
        self.conv_block6 = ConvBlock(128, 128, 3)
        if self.attention:
            
            self.attn1 = AttentionBlock(in_features_l=128, in_features_g=128, attn_features=4, up_factor=4, normalize_attn=normalize_attn)
            self.attn2 = AttentionBlock(in_features_l=128, in_features_g=128, attn_features=4, up_factor=2, normalize_attn=normalize_attn)
            self.attn3 = AttentionBlock(in_features_l=128, in_features_g=128, attn_features=4, up_factor=1, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify1 = nn.Sequential(
			                    nn.Linear(in_features=128, out_features=num_classes*2, bias=True))
            self.classify2 = nn.Sequential(
			                    nn.Linear(in_features=128, out_features=num_classes*2, bias=True))
            self.classify3 = nn.Sequential(
			                    nn.Linear(in_features=128, out_features=num_classes*2, bias=True))
        else:
            self.classify = nn.Linear(in_features=128, out_features=num_classes, bias=True)


    def forward(self, x, return_att = False):
        # feed forward
        x = self.conv_block1(x)
        l1 = self.conv_block3(x) # /1
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0) # /2
        l2 = self.conv_block4(x) # /2
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0) # /4
        l3 = self.conv_block5(x) # /4
        x = self.conv_block6(l3) # /32
        g = x 
        # Pay attention
        if self.attention:
            c1, g1 = self.attn1(l1, g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            
            x_1 = self.classify1(g1)
            x_2 = self.classify2(g2)
            x_3 = self.classify3(g3)
            x = (x_1+x_2+x_3)/3
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
            
        mu, log_std = x.chunk(2, dim=1)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        if return_att:
          return [dist, c1, c2, c3]
         
        return dist


    def visual_att(self, train_loader, device, args):
        bat = next(iter(train_loader))[0].to(device)
        bat_ = add_noise(bat, args.noise_level)
        [x, c1, c2, c3] = self.forward(bat_, return_att = True)
        plot_att(x,c1,c2,c3, bat, args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level), args.norm_att)

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.95, N+4)
    return mycmap

def my_norm(x):
  weigghts = torch.arange(4096).repeat(8).view(-1,4096)
  weigghts[0, 0:1570] = 0
  weigghts[0, 1571:4096] = torch.arange(2525)
  weigghts[-1, 1571:4096] = 0
  return ((weigghts)*0.000085)-0.22+ x
  
def plot_att(x,c1,c2,c3, batch, title, norm_att):
  wavelengths = genfromtxt("../data/wl_solution.txt", delimiter=',')
  wavelengths = torch.from_numpy(wavelengths)
  plt.rcParams['figure.figsize'] = [15, 5]
  mycmap = transparent_cmap(plt.cm.jet)

  N,C,W,H = c1.size()
  if norm_att:
    c1 = F.softmax(c1.view(N,C,-1), dim=2).view(N,C,W,H)
  else:
    c1 = torch.sigmoid(c1)
  N,C,W,H = c2.size()
  if norm_att:
    c2 = F.softmax(c2.view(N,C,-1), dim=2).view(N,C,W,H)
  else:
    c2 = torch.sigmoid(c2)
  N,C,W,H = c3.size()
  if norm_att:
    c3 = F.softmax(c3.view(N,C,-1), dim=2).view(N,C,W,H)
  else:
    c3 = torch.sigmoid(c3)

  att = F.interpolate(c1, scale_factor=1, mode='bilinear', align_corners=False)
  att1 = F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)
  att2 = F.interpolate(c3, scale_factor=4, mode='bilinear', align_corners=False)
  
  att = att.view(att.size(0),-1) 
  att -= att.min(1,keepdim=True)[0]
  att /= att.max(1,keepdim=True)[0]
  att = att.view(128,1,8,4096)
  
  att1 = att1.view(att1.size(0),-1) 
  att1 -= att1.min(1,keepdim=True)[0]
  att1 /= att1.max(1,keepdim=True)[0]
  att1 = att1.view(128,1,8,4096)
  
  att2 = att2.view(att2.size(0),-1) 
  att2 -= att2.min(1,keepdim=True)[0]
  att2 /= att2.max(1,keepdim=True)[0]
  att2 = att2.view(128,1,8,4096)
  
  fig,a =  plt.subplots(2,1)
  #a[0,0].plot(
  ids = random.randint(0,128)
  first_norm = my_norm(batch[ids].cpu())
  a[0].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), first_norm.view(-1,8*4096).squeeze(0).numpy())
  a[1].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), att[ids].view(-1,8*4096).squeeze(0).cpu().detach().numpy(),'b', label = 'Attention block 1')
  a[1].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), att1[ids].view(-1,8*4096).squeeze(0).cpu().detach().numpy(),'r', label = 'Attention block 2')
  a[1].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), att2[ids].view(-1,8*4096).squeeze(0).cpu().detach().numpy(),'g', label = 'Attention block 3')
  a[1].legend(loc='upper left')
  plt.savefig(title+"1.png")
  plt.clf()
  fig,a =  plt.subplots(2,1)
  ids = random.randint(0,128)
  first_norm = my_norm(batch[ids].cpu())
  a[0].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), first_norm.view(-1,8*4096).squeeze(0).numpy())
  a[1].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), att[ids].view(-1,8*4096).squeeze(0).cpu().detach().numpy(),'b', label = 'Attention block 1')
  a[1].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), att1[ids].view(-1,8*4096).squeeze(0).cpu().detach().numpy(),'r', label = 'Attention block 2')
  a[1].plot(wavelengths.view(-1,8*4096).squeeze(0).numpy(), att2[ids].view(-1,8*4096).squeeze(0).cpu().detach().numpy(),'g', label = 'Attention block 3')
  a[1].legend(loc='upper left')
  plt.savefig(title+"2.png")

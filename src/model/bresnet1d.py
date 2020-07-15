import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet1d import ResnetBlock1d, Flatten

class BayesianResidualNetworkD1(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 input_channels = 1,
                 convolutions = [16,16],
                 kernel_size = [7,7],
                 hiddenlayer=[512,256],
                 maxpool = 4,
                 dropout = 0.1):
        super(BayesianResidualNetworkD1, self).__init__()


        convLayers = []
        self.out_size = out_size
        
        ## Convolutional layers
        for i, convolution  in enumerate(convolutions):
            if (i==0):
                convLayers.append(nn.Conv1d(in_channels = input_channels, out_channels = convolutions[0],
                                padding = kernel_size[0]//2, kernel_size = kernel_size[0]))
                convLayers.append(nn.BatchNorm1d(convolutions[0]))
                #layers.append(nn.MaxPool1d(maxpool))
                convLayers.append(nn.ReLU())
                #convLayers.append(nn.Dropout(p=dropout))          
            else:
                convLayers.append(ResnetBlock1d(in_num_filter = convolutions[i-1], out_num_filters = convolutions[i],
                                padding_size = kernel_size[i]//2, kernel_sizes = kernel_size[i]))
                #convLayers.append(nn.BatchNorm1d(convolutions[i]))
                convLayers.append(nn.AvgPool1d(maxpool))
                convLayers.append(nn.ReLU())
                convLayers.append(nn.Dropout(p=dropout))
                
        # Flatten before fully connected
        #layers.append(nn.MaxPool1d(maxpool))
        
        flatten_layers = []
        flatten_layers.append(Flatten())
        
        ## Fully Connected layers
        for i, layer in enumerate(hiddenlayer):
            if i == 0:
                # TODO: when u add maxpool
                flatten_layers.append(nn.Linear(in_size//(maxpool**(len(convolutions)-1))*convolutions[-1],hiddenlayer[i]))
                #flatten_layers.append(nn.Linear(int(in_size*convolutions[-1]),hiddenlayer[i]))
                #flatten_layers.append(nn.BatchNorm1d(hiddenlayer[i]))
                flatten_layers.append(nn.ReLU())
                flatten_layers.append(nn.Dropout(p=dropout))
                
            else:
                flatten_layers.append(nn.Linear(hiddenlayer[i-1], hiddenlayer[i]))
                #flatten_layers.append(nn.BatchNorm1d(hiddenlayer[i]))
                flatten_layers.append(nn.ReLU())
                flatten_layers.append(nn.Dropout(p=dropout))
                
        ## Remove last fully connected
        flatten_layers = flatten_layers[:-1]
        flatten_layers.append(nn.Linear(hiddenlayer[-1], (4+4)))
        self.convnet = nn.Sequential(*convLayers)
        self.flatnet = nn.Sequential(*flatten_layers)
        
        
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.convnet(x)
        out = self.flatnet(x)
        mu, log_std = out.chunk(2, dim=1)
        #loc, tril, diag = out.split([4,6,4], dim=-1)
        #diag = 1 + nn.functional.elu(diag)
        #z = torch.zeros(size=[loc.size(0)], device=out.device)
        #scale_tril = torch.stack([
        #	diag[:, 0], z ,   z,  z,
        #	tril[:, 0], diag[:, 1], z, z,
        #	tril[:, 1], tril[:, 3], diag[:, 2],z,
        #    tril[:, 2], tril[:, 4],  tril[:,5], diag[:, 3]
        #], dim=-1).view(-1, 4, 4)
		# scale_tril is a tensor of size [batch, 3, 3]
        #dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        return dist
        

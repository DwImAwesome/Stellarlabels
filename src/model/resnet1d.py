import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResnetBlock1d(nn.Module):
    def __init__(self, in_num_filter, out_num_filters, padding_size, kernel_sizes):
        super(ResnetBlock1d, self).__init__()
        
        self.residual = nn.Sequential(
                    nn.Conv1d(in_num_filter, in_num_filter, kernel_size=kernel_sizes, padding=padding_size),
                    nn.BatchNorm1d(in_num_filter),
                    nn.ReLU(),
                    nn.Conv1d(in_num_filter, in_num_filter, kernel_size=5, padding=2),
                    nn.BatchNorm1d(in_num_filter)
                    )

    
    def forward(self, x):
        return self.residual(x)+ x


class ResidualNetworkD1(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 input_channels = 1,
                 convolutions = [16,16],
                 kernel_size = [7,7],
                 hiddenlayer=[512,256],
                 maxpool = 4,
                 dropout = 0.1):
        super(ResidualNetworkD1, self).__init__()


        convLayers = []
        
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
                flatten_layers.append(nn.Linear(in_size//(maxpool**(len(convolutions)-1))*convolutions[-1],hiddenlayer[i]))
                flatten_layers.append(nn.ReLU())
                flatten_layers.append(nn.Dropout(p=dropout))
                
            else:
                flatten_layers.append(nn.Linear(hiddenlayer[i-1], hiddenlayer[i]))
                flatten_layers.append(nn.ReLU())
                flatten_layers.append(nn.Dropout(p=dropout))
        
        flatten_layers.append(nn.Linear(hiddenlayer[-1], out_size))
        self.convnet = nn.Sequential(*convLayers)
        self.flatnet = nn.Sequential(*flatten_layers)
        
        
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.convnet(x)
        out = self.flatnet(x)
        x = x.unsqueeze(1)
        return out
        

from IPython.core.completerlib import module_list
import torch
import torch.nn as nn
import torch.nn.functional as F


#define image size 50x50 pixels
img_size = 100



class TunableNet(nn.Module):

    def __init__(self, channels, fc_units, num_pooling, dropout2d, dropout, kernel_size, adaptivepool, pooling2d=2):

        # call constructor method from parent class
        super().__init__()

        self.convolutional_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        

        in_channels = 3

        for i, out_channels in enumerate(channels):

            self.convolutional_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(nn.BatchNorm2d(out_channels))
            self.pooling_layers.append(nn.MaxPool2d(pooling2d) if i < num_pooling else nn.Identity())
            self.dropout_layers.append(nn.Dropout2d(dropout2d))

            in_channels = out_channels


        
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptivepool)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(in_channels * adaptivepool * adaptivepool, fc_units)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 2)




    def forward(self, x):

        for conv, norm, pool, drop in zip(self.convolutional_layers, self.norm_layers, self.pooling_layers, self.dropout_layers):
            x = conv(x)
            x = norm(x)
            x = F.relu(x)
            x = pool(x)
            x = drop(x)

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        

        


        return x

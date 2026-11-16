import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


#define image size 50x50 pixels
img_size = 100



class Net(nn.Module):

    def __init__(self):

        # call constructor method from parent class
        super().__init__()
        
        #convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(32) #batch normalization

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        self.dropout_2d = nn.Dropout2d(p=0.25) #channel dropout
        self.norm2 = nn.BatchNorm2d(64) #batch normalization
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7)

        self.maxpool = nn.MaxPool2d((2,2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((9, 9))

        #fully connected layers
        self.fc1 = nn.Linear(128*9*9, 512)
        #unit droupout
        self.dropout_ind = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 2)



    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        #print(f"shape after conv1 {x.shape}")

        x = self.dropout_2d(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        #print(f"shape after conv2 {x.shape}")

        x = self.dropout_2d(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.adaptive_pool(x)
        #print(f"shape after conv3 {x.shape}")

        


        #sys.exit("getting shape of input for fully connected layer")

        x = x.view(-1, 128*9*9)

        x = F.relu(self.fc1(x))

        x = self.dropout_ind(x)

        x = self.fc2(x)
        #x = F.softmax(x)

        return x


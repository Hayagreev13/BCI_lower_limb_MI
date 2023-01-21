""" Feature Extractor EEGNet model definition - pretrain model"""

import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FeatureExtractor(nn.Module):

    def __init__(self, config, mtl=True): #add parameters here wandb and optimization
        super(FeatureExtractor, self).__init__()
        self.config = config
        receptive_field = 64
        filter_sizing = 8
        dropout = 0.1
        D = 2
        sample_duration = 500
        channel_amount = 8
        if self.config['clstype'] == 'legs':
            num_classes = 2
        elif self.config['clstype'] == 'multiclass':
            num_classes = 4  
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ReLU(True),
        )
        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ReLU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 4], stride=[1, 4], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 8], stride=[1, 8], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = filter_sizing*D*15

    def forward(self,x):
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        out = self.seperable(out)
        out = self.avgpool2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        #output shape of out = [batch_size, 208] #240 in new model
        #using out instead of prediction to adapt it to MUPS code
        #prediction = self.fc2(out)
        #output shape of prediction = [batch_size,4]
        return out

class MindReader(nn.Module):

    def __init__(self, mtl=True): #add parameters here wandb and optimization
        super(MindReader, self).__init__()
        receptive_field = 64
        filter_sizing = 8
        dropout = 0.1
        D = 2
        channel_amount = 8
        num_classes = 2
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ReLU(True),
        )
        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ReLU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 4], stride=[1, 4], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 8], stride=[1, 8], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = filter_sizing*D*15
        self.fc2 = nn.Linear(endsize, num_classes)

    def forward(self,x):
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        out = self.seperable(out)
        out = self.avgpool2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        #output shape of out = [batch_size, 208] #240 in new model
        #using out instead of prediction to adapt it to MUPS code
        prediction = self.fc2(out)
        #output shape of prediction = [batch_size,4]
        return prediction
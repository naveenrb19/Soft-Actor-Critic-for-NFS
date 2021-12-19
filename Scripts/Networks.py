import torch.nn as nn
import torch.functional as F
import torch
class Actor(nn.Module):
    def __init__(self,img_ch,feat,num_actions):
        super(Actor,self).__init__()
        
        #640x640x3
        self.conv1=nn.Sequential(
            self.SeperableConv2D(in_channels=img_ch,out_channels=feat,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.3))
            #320x320x16
        self.convbatch=nn.Sequential(
                self.ConvBlockBatch(feat,feat*2,4,2,1),
                #160x160x32
                self.ConvBlockBatch(feat*2,feat*4,4,2,1),
                #80x80x64
                self.ConvBlockBatch(feat*4,feat*8,4,2,1),
                #40x40x128
                self.ConvBlockBatch(feat*8,feat*16,4,2,1),
                #20x20x256
                self.ConvBlockBatch(feat*16,feat*32,4,2,1),
                #10x10x512
                self.ConvBlockBatch(feat*32,feat*32,4,2,1),
                #5x5x512
                self.ConvBlockBatch(feat*32,feat*64,5,1,0),
            )
        self.convnobatch=nn.Sequential(
                self.ConvBlockNoBatch(feat,feat*2,4,2,1),
                #160x160x32
                self.ConvBlockNoBatch(feat*2,feat*4,4,2,1),
                #80x80x64
                self.ConvBlockNoBatch(feat*4,feat*8,4,2,1),
                #40x40x128
                self.ConvBlockNoBatch(feat*8,feat*16,4,2,1),
                #20x20x256
                self.ConvBlockNoBatch(feat*16,feat*32,4,2,1),
                #10x10x512
                self.ConvBlockNoBatch(feat*32,feat*32,4,2,1),
                #5x5x512
                self.ConvBlockNoBatch(feat*32,feat*64,5,1,0),
            )
        self.relu=nn.LeakyReLU(0.2)
        self.fc1=nn.Linear(feat*64,feat*32)
        self.mu=nn.Linear(feat*32,num_actions)
        self.log_std=nn.Linear(feat*32,num_actions)
        
        
    def ConvBlockBatch(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            self.SeperableConv2D(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3)
        )
    def ConvBlockNoBatch(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            self.SeperableConv2D(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.LeakyReLU(0.3)
        )
    def SeperableConv2D(self,in_channels,out_channels,kernel_size,stride,padding,groups=10,bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,stride=stride,groups=in_channels),
            nn.Conv2d(in_channels,out_channels,kernel_size=1)
        )
    def forward(self,x):
        x=self.conv1(x)
        bs=x.shape[0]
        x=self.convnobatch(x)
        x=x.view(bs,-1)
        x=self.fc1(x)
        x=self.relu(x)
        mu=self.mu(x)
        log_std=self.log_std(x)
        log_std=torch.clamp(log_std,-1,1)
        return mu,log_std
class Critic(nn.Module):
    def __init__(self,img_ch,feat,num_actions):
        super(Critic,self).__init__()
        self.conv1=nn.Sequential(
            self.SeperableConv2D(in_channels=img_ch,out_channels=feat,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.3))
            #320x320x16
        self.convbatch=nn.Sequential(
                self.ConvBlockBatch(feat,feat*2,4,2,1),
                #160x160x32
                self.ConvBlockBatch(feat*2,feat*4,4,2,1),
                #80x80x64
                self.ConvBlockBatch(feat*4,feat*8,4,2,1),
                #40x40x128
                self.ConvBlockBatch(feat*8,feat*16,4,2,1),
                #20x20x256
                self.ConvBlockBatch(feat*16,feat*32,4,2,1),
                #10x10x512
                self.ConvBlockBatch(feat*32,feat*32,4,2,1),
                #5x5x512
                self.ConvBlockBatch(feat*32,feat*64,5,1,0),
            )
        self.convnobatch=nn.Sequential(
                self.ConvBlockNoBatch(feat,feat*2,4,2,1),
                #160x160x32
                self.ConvBlockNoBatch(feat*2,feat*4,4,2,1),
                #80x80x64
                self.ConvBlockNoBatch(feat*4,feat*8,4,2,1),
                #40x40x128
                self.ConvBlockNoBatch(feat*8,feat*16,4,2,1),
                #20x20x256
                self.ConvBlockNoBatch(feat*16,feat*32,4,2,1),
                #10x10x512
                self.ConvBlockNoBatch(feat*32,feat*32,4,2,1),
                #5x5x512
                self.ConvBlockNoBatch(feat*32,feat*64,5,1,0),
            )
        self.relu=nn.LeakyReLU(0.2)
        self.fcstate=nn.Linear(feat*64,feat*32)
        self.fcact=nn.Linear(num_actions,feat*32)
        self.fc=nn.Linear(1024,1)
        self.relu=nn.LeakyReLU(0.2)
        
        
    def ConvBlockBatch(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            self.SeperableConv2D(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3)
        )
    def ConvBlockNoBatch(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            self.SeperableConv2D(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.LeakyReLU(0.3)
        )
    def SeperableConv2D(self,in_channels,out_channels,kernel_size,stride,padding,groups=10,bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,stride=stride,groups=in_channels),
            nn.Conv2d(in_channels,out_channels,kernel_size=1))
    def forward(self,x,a):
        x=self.conv1(x)
        bs=x.shape[0]
        x=self.convnobatch(x)
        x=x.view(bs,-1)
        # print(x.shape)
        state=self.fcstate(x)
        state=self.relu(state)
        act=self.fcact(a)
        act=self.relu(act)
        # print(state.shape)
        # print(act.shape)
        tot=torch.cat((state,act),1)
        # print(tot.shape)
        res=self.fc(tot)
        return res
class ValueNetwork(nn.Module):
    def __init__(self,img_ch,feat):
        super(ValueNetwork,self).__init__()
        self.conv1=nn.Sequential(
            self.SeperableConv2D(in_channels=img_ch,out_channels=feat,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.3))
            #320x320x16
        self.convbatch=nn.Sequential(
                self.ConvBlockBatch(feat,feat*2,4,2,1),
                #160x160x32
                self.ConvBlockBatch(feat*2,feat*4,4,2,1),
                #80x80x64
                self.ConvBlockBatch(feat*4,feat*8,4,2,1),
                #40x40x128
                self.ConvBlockBatch(feat*8,feat*16,4,2,1),
                #20x20x256
                self.ConvBlockBatch(feat*16,feat*32,4,2,1),
                #10x10x512
                self.ConvBlockBatch(feat*32,feat*32,4,2,1),
                #5x5x512
                self.ConvBlockBatch(feat*32,feat*64,5,1,0),
            )
        self.convnobatch=nn.Sequential(
                self.ConvBlockNoBatch(feat,feat*2,4,2,1),
                #160x160x32
                self.ConvBlockNoBatch(feat*2,feat*4,4,2,1),
                #80x80x64
                self.ConvBlockNoBatch(feat*4,feat*8,4,2,1),
                #40x40x128
                self.ConvBlockNoBatch(feat*8,feat*16,4,2,1),
                #20x20x256
                self.ConvBlockNoBatch(feat*16,feat*32,4,2,1),
                #10x10x512
                self.ConvBlockNoBatch(feat*32,feat*32,4,2,1),
                #5x5x512
                self.ConvBlockNoBatch(feat*32,feat*64,5,1,0),
            )
        self.relu=nn.LeakyReLU(0.2)
        self.fc1=nn.Linear(feat*64,feat*32)
        self.fcf=nn.Linear(feat*32,1)
        
    def SeperableConv2D(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,stride=stride,groups=in_channels),
            nn.Conv2d(in_channels,out_channels,kernel_size=1))    
        
    def ConvBlockBatch(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            self.SeperableConv2D(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3)
        )
    def ConvBlockNoBatch(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            self.SeperableConv2D(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.LeakyReLU(0.3)
        )
    def forward(self,x):
        x=self.conv1(x)
        bs=x.shape[0]
        x=self.convnobatch(x)
        x=x.view(bs,-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fcf(x)
        return x
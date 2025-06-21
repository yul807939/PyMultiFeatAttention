
import torch

class SE_block(torch.nn.Module):
    def __init__(self,inchannel,ratio = 4):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Sequential(
            torch.nn.Linear(inchannel,inchannel // ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(inchannel // ratio,inchannel),
            torch.nn.Hardsigmoid(inplace=True)
        )

    def forward(self,input):
        b,c,_,_ = input.size()
        x = self.pool(input)
        x = x.view([b,c])
        x = self.conv(x)
        x = x.view([b,c,1,1])
        return x*input

class MB_block(torch.nn.Module):
    def __init__(self,input_channels,outchannel,kernal,stride,expand_ratio):
        super().__init__()
        self.input_channels = input_channels
        self.outchannel = outchannel
        self.stride = stride
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,input_channels*expand_ratio,1),
            torch.nn.BatchNorm2d(input_channels*expand_ratio),
            torch.nn.Hardswish(),
            torch.nn.Conv2d(input_channels*expand_ratio,input_channels*expand_ratio,kernal,stride,padding=kernal // 2,groups=input_channels*expand_ratio),
            torch.nn.BatchNorm2d(input_channels*expand_ratio),
            torch.nn.Hardswish(),
            SE_block(input_channels*expand_ratio),
            torch.nn.Conv2d(input_channels*expand_ratio,outchannel,1,1),
            torch.nn.BatchNorm2d(outchannel)
        )
    def forward(self,x):
        out = self.conv(x)
        if self.stride == 1 and self.input_channels == self.outchannel:
            out += x
        return out

class EfficientNet(torch.nn.Module):
    def __init__(self,input_channels,classes):
        super().__init__()
        self.input_channels = input_channels
        self.classes = classes
        self.stage_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardswish()
        )
        self.stage_2 = torch.nn.Sequential(
            MB_block(32,16,3,1,1)
        )
        self.stage_3 = torch.nn.Sequential(
            MB_block(16,24,3,2,6),
            MB_block(24,24,3,1,6)
        )
        self.stage_4 = torch.nn.Sequential(
            MB_block(24,40,5,2,6),
            MB_block(40,40,5,1,6)
        )
        self.stage_5 = torch.nn.Sequential(
            MB_block(40,80,3,2,6),
            MB_block(80,80,3,1,6),
            MB_block(80,80,3,1,6),
        )
        self.stage_6 = torch.nn.Sequential(
            MB_block(80,112,5,1,6),
            MB_block(112,112,5,1,6),
            MB_block(112,112,5,1,6),
        )
        self.stage_7 = torch.nn.Sequential(
            MB_block(112,192,5,2,6),
            MB_block(192,192,5,1,6),
            MB_block(192,192,5,1,6),
            MB_block(192,192,5,1,6),
        )
        self.stage_8 = torch.nn.Sequential(
            MB_block(192,320,3,1,6)
        )
        self.stage_9 = torch.nn.Sequential(
            torch.nn.Conv2d(320,1280,1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(1280,classes)
        )
    def forward(self,x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.stage_5(x)
        x = self.stage_6(x)
        x = self.stage_7(x)
        x = self.stage_8(x)
        x = self.stage_9(x)
        return x





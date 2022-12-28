import torch
from torch import nn
from typing import Union

def get_activation(activation:str, neg_slope:float=0.1):
    if activation.lower() not in {'relu', 'relu6', 'leaky', 'elu', 'selu', 'celu', 'linear'}:
        raise NameError("Make sure activation is one of 'relu', 'relu6', 'leaky', 'elu', 'selu', 'celu','linear'")
    act = activation.lower()
    if act   == 'relu': act = nn.ReLU(inplace=True)
    elif act =='relu6': act = nn.ReLU6(inplace=True)
    elif act =='leaky': act = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
    elif act == 'elu' : act = nn.ELU(alpha=1,inplace=True)
    elif act == 'selu': act = nn.SELU(inplace=True)
    elif act == 'celu': act = nn.CELU(alpha=1,inplace=True)
    elif act=='linear': act = None
    return act

class ConvBlock(nn.Module):
    def __init__(self,
                in_features:int, 
                out_features:int, 
                kernel_size:int, 
                strides:int=1,
                padding:Union[int, str] = 'same',
                activation:str='relu',
                neg_slope:float=0.1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, strides, padding)
        self.bn   = nn.BatchNorm2d(out_features)
        self.act = get_activation(activation, neg_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act: x = self.act(x)
        return x

class ResidualBottleneck(nn.Module):
    def __init__(self, 
                in_features:int, 
                reduction:float=0.5,
                activation:str='relu',
                neg_slope:float=0.1) -> None:
        super().__init__()
        hidden_features = int(in_features*reduction)
        self.conv1 = ConvBlock(in_features, hidden_features, 1, activation='leaky')
        self.conv2 = ConvBlock(hidden_features, in_features, 3, activation='linear')
        self.act = get_activation(activation, neg_slope)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        x = self.act(x)
        return x
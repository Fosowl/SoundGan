import torch
import torch.nn as nn
import math

def conv_output_size(input_size, kernel_size, stride, padding):
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        ks = 4
        stride = 2
        pad = 1
        w, h = config.image_size
        
        self.conv1 = nn.Conv2d(config.nc, config.ndf, kernel_size=ks, stride=stride, padding=pad, bias=False)
        w, h = conv_output_size(w, ks, stride, pad), conv_output_size(h, ks, stride, pad)
        self.relu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(config.ndf, config.ndf * 2, kernel_size=ks, stride=stride, padding=pad, bias=False)
        w, h = conv_output_size(w, ks, stride, pad), conv_output_size(h, ks, stride, pad)
        self.relu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(config.ndf * 2, config.ndf * 4, kernel_size=ks, stride=stride, padding=pad, bias=False)
        w, h = conv_output_size(w, ks, stride, pad), conv_output_size(h, ks, stride, pad)
        self.relu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(config.ndf * 4, config.ndf * 8, kernel_size=ks, stride=stride, padding=pad, bias=False)
        w, h = conv_output_size(w, ks, stride, pad), conv_output_size(h, ks, stride, pad)
        self.relu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(config.ndf * 8, config.ndf * 16, kernel_size=ks, stride=stride, padding=pad, bias=False)
        w, h = conv_output_size(w, ks, stride, pad), conv_output_size(h, ks, stride, pad)
        self.relu5 = nn.LeakyReLU(0.2)
        
        self.conv6 = nn.Conv2d(config.ndf * 16, 1, kernel_size=ks, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv6(x)
        x = self.flatten(x)
        return x
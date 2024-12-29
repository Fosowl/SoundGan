import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        ks = 4
        stride = 2
        pad = 1
        
        self.conv1 = nn.ConvTranspose2d(config.nz, config.ngf * 32, kernel_size=4, stride=1, padding=0, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.ConvTranspose2d(config.ngf * 32, config.ngf * 16, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.ConvTranspose2d(config.ngf * 16, config.ngf * 8, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.ConvTranspose2d(config.ngf * 8, config.ngf * 4, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.ConvTranspose2d(config.ngf * 4, config.ngf * 2, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.ConvTranspose2d(config.ngf * 2, config.ngf, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv7 = nn.ConvTranspose2d(config.ngf, config.nc, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(z))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.tanh(self.conv7(x))
        return x
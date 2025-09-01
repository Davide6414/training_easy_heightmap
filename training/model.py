import torch, torch.nn as nn, torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, ks=3, s=1, p=1, gn=8):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, ks, s, p)
        self.n = nn.GroupNorm(gn, cout)
        self.r = nn.ReLU(inplace=True)
    def forward(self, x): return self.r(self.n(self.c(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBNReLU(c, c)
        self.c2 = ConvBNReLU(c, c)
    def forward(self, x): 
        r = self.c2(self.c1(x))
        return x + r

class SRUNet(nn.Module):
    def __init__(self, base=48, depth=4):
        super().__init__()
        self.stem = ConvBNReLU(1, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base*4, base*4))
        self.up2   = nn.Sequential(nn.ConvTranspose2d(base*4, base*2, 2, 2), ConvBNReLU(base*2, base*2))
        self.up1   = nn.Sequential(nn.ConvTranspose2d(base*2, base, 2, 2),   ConvBNReLU(base, base))
        self.head  = nn.Conv2d(base, 1, 3, 1, 1)

        self.rb = nn.Sequential(*[Block(base*4) for _ in range(4)])

    def forward(self, low):
        x0 = self.stem(low)       # 256
        x1 = self.down1(x0)       # 128
        x2 = self.down2(x1)       # 64
        x3 = self.down3(x2)       # 32
        x3 = self.rb(x3)
        u2 = self.up2(x3) + x1
        u1 = self.up1(u2) + x0
        resid = self.head(u1)
        return resid

import torch
import torch.nn as nn

class conv2d_bn(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size == 3:
            self.conv = nn.Conv2d(4,4,kernel_size=kernel_size,stride=1,padding=1) 
        elif kernel_size == 1:
            self.conv = nn.Conv2d(4,4,kernel_size=kernel_size,stride=1) 
        self.bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Inception(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.conv1=conv2d_bn(kernel_size=1)
        
        self.conv2a=conv2d_bn(kernel_size=1)
        self.conv2b=conv2d_bn(kernel_size=3)
        
        self.avg_pool3=nn.AvgPool2d(kernel_size=3,stride=1, padding=1)
        self.conv3=conv2d_bn(kernel_size=3)
        
        self.conv4a=conv2d_bn(kernel_size=1)
        self.conv4b=conv2d_bn(kernel_size=3)
        self.conv4c=conv2d_bn(kernel_size=3)
        
        input_size = h*w*4
        output_size = h*w
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self,x):
        x1=self.conv1(x)
        print(f"x1.shape:{x1.shape}")
        
        x2=self.conv2a(x)
        x2=self.conv2b(x2)
        print(f"x2.shape:{x2.shape}")
        
        x3=self.avg_pool3(x)
        x3= self.conv3(x3)
        print(f"x3.shape:{x3.shape}")
        
        x4 = self.conv4a(x)
        x4 = self.conv4b(x4)
        x4 = self.conv4c(x4)
        print(f"x4.shape:{x4.shape}")
        
        b, c, h, w = x.shape
        x1 = x1.view(b, c,-1)
        x2 = x2.view(b, c,-1)
        x3 = x3.view(b, c,-1)
        x4 = x4.view(b, c,-1)
        
        x_cat = torch.cat((x1,x2,x3,x4), dim = -1)
        output = self.linear(x_cat).view(b,c,h,w)
        return output
    
if __name__ == '__main__':
    icp = Inception(32,32)
    x = torch.randn(32,4,32,32)
    print(f"x.shape:{x.shape}")
    output = icp(x)
    print(f"output.shape:{output.shape}")
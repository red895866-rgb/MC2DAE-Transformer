import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Multi_Chan_Conv(nn.Module):
    def __init__(self):
        super(Multi_Chan_Conv, self).__init__()

        self.conv1d_1 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv1d_2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv1d_3 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv1d_4 = nn.Conv1d(320, 512, kernel_size=3, stride=2, padding=1)

        self.conv1d_b = nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2)

        self.chan_8_conv1d_1 = nn.ModuleList([nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1) for i in range(8)])
        self.chan_8_conv1d_2 = nn.ModuleList([nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1) for i in range(8)])

        self.chan_8_conv2d_1 = nn.ModuleList([nn.Conv2d(1, 4, kernel_size=(3,3), stride=(2,2), padding=(1,1)) for i in range(8)])
        self.chan_8_conv2d_2 = nn.ModuleList([nn.Conv2d(4, 8, kernel_size=(4,5), stride=(4,4), padding=(0,2)) for i in range(8)])

        self.deconv1d_1 = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1) 
        self.deconv1d_2 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.deconv1d_3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, padding=1,output_padding=1)  
        self.deconv1d_4 = nn.ConvTranspose1d(256, 64, kernel_size=2, stride=2)  
        
    
        self.act = nn.PReLU()
                
    def forward(self, x):  
        x_64 = x.reshape(-1, 64, 550)
        h_1 = self.conv1d_1(x_64)
        h_1 = self.act(h_1)
        h_2 = self.conv1d_2(h_1) 
        h_2 = self.act(h_2)
        
        h_3 = self.conv1d_b(x_64) 
        h_3 = self.act(h_3) 
        
        h_1d3 = [self.chan_8_conv1d_1[i](x[:,:,i,:]) for i in range(8)] 
        h_1d3 = [self.act(h_1d3[i]) for i in range(8)]
        h_1d3 = [self.chan_8_conv1d_2[i](h_1d3[i]) for i in range(8)]
        h_1d3 = [self.act(h_1d3[i]) for i in range(8)]
        h_1d3 = torch.cat(h_1d3,dim=1) 
        
        h_4 = torch.cat([h_2,h_3,h_1d3],dim=1) 
        h_4 = self.conv1d_3(h_4) 
        h_4 = self.act(h_4) 
        
        h_2d1 = [self.chan_8_conv2d_1[i](x[:,:,i:i+1,:].transpose(2,1)) for i in range(8)]
        h_2d1 = [self.act(h_2d1[i]) for i in range(8)]
        h_2d1 = [self.chan_8_conv2d_2[i](h_2d1[i]).squeeze(dim=2) for i in range(8)]
        h_2d1 = [self.act(h_2d1[i]) for i in range(8)]
        h_2d1 = torch.cat(h_2d1,dim=1) 
        
        h_5 = torch.cat([h_4,h_2d1],dim=1)
        h_5 = self.conv1d_4(h_5)
        h_5 = self.act(h_5)
    
        de_out = self.deconv1d_1(h_5) 
        de_out = self.act(de_out)       
        de_out = self.deconv1d_2(de_out)        
        de_out = self.act(de_out)
        de_out = torch.cat([de_out,h_2],dim=1)
        de_out = self.deconv1d_3(de_out)
        de_out = self.act(de_out)
        de_out = torch.cat([de_out,h_1],dim=1)
        de_out = self.deconv1d_4(de_out)
       
        return de_out

if __name__ == "__main__":
    model = Multi_Chan_Conv().to("cuda")
    input_tensor = torch.randn(32, 8,8, 550).to("cuda") 

    o1 = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    o1.shape
    
    
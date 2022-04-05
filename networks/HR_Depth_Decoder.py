from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from hr_layers import *
from layers import upsample
from my_utils import plotTensorMultiple, toNumpy, to_tensor
from utils import estimateA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
        
        # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else: 
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
            self.convs["72"] = Attention_Module(self.num_ch_enc[4]  , self.num_ch_enc[3] * 2, 256)
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128)
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64 , 64)
            self.convs["9"] = Attention_Module(64, 64, 32)
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72 , feature36)
        x18 = self.convs["18"](x36 , feature18)
        x9 = self.convs["9"](x18,[feature64])
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))

        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        return outputs
        

## BGR_Attetion
class BG_R_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(BG_R_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.Linear(in_planes // ratio, 1, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(x.view(b, 3,-1))
        out = 2*self.sigmoid(avg_out)-1
        return out


class WaterTypeRegression(nn.Module):
    def __init__(self, in_channel):
        super(WaterTypeRegression, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, padding = 1),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(16, 8, 5, padding = 1, stride = 2),
            nn.Flatten(),
            nn.Linear(151368, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
            )       

    def print_sizes(self, model, input_tensor):
        output = input_tensor
        for m in model.children():
            output = m(output)
            print(m, output.shape)
        return output 

    def forward(self, x, d):
        """ A forward pass of your neural net (evaluates f(x)).
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        # input_tensor = x.clone()
        # # self.print_sizes(self.model, input_tensor)
        # #Normalise in forward from fit()
        # mean = torch.mean(input_tensor)
        # std = torch.std(input_tensor)
        # input_tensor = (input_tensor - mean) / std

        # x = x.reshape(-1, 3, 32, 32)
        wt_coeffs = self.model(x).view(-1, 3, 1, 1)
        TM = torch.exp(-wt_coeffs*d)
        batch_size = x.shape[0]
        A = torch.zeros(1,3)
        for i in range(batch_size):
            img = toNumpy(x[i,:,:,:])
            depth = toNumpy(d[i,:,:])
            Ai = torch.from_numpy(estimateA(img, depth))
            A+=Ai
        A/=batch_size
        # A = torch.tensor((0.5020, 0.7941, 0.4000))

        A = torch.repeat_interleave(A.view(1,3,1,1),batch_size, dim=0).to(device)

        # print(A)
        J = (x - A) / TM + A
        return J, wt_coeffs


class BG2RCoeffsNetwork(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BG2RCoeffsNetwork, self).__init__()
        
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()
        self.cs = CS_Block(in_channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1 )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(240*320, 3)
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()
        self.bgrAtt = BG_R_Attention(76800, 16)

    def forward(self, input_features, image):
        features = [upsample(input_features)]
    
        features = torch.cat(features, 1)

        features = self.ca(features)
        features = self.sa(features)
        features = self.cs(features)
        
        mu_vec = self.conv_se(features)
        mu_vec = self.bgrAtt(mu_vec)
        mu_vec = torch.unsqueeze(mu_vec, dim=3)
        R = torch.unsqueeze(image[:,0,:,:], dim=1)
        BG = torch.max(image[:,1:, :,:], dim=1, keepdim=True)[0]
        ones = torch.ones_like(R)
        # BG_R = torch.max(image[:,1:, :,:], dim=1, keepdim=True)[0] - torch.unsqueeze(image[:,0,:,:], dim=1)
        BG_R = torch.squeeze(torch.stack((ones, BG, R), dim=1), dim=2)
        depth = torch.sum(mu_vec*BG_R, dim=1, keepdim=True)
        depth = torch.max(depth,torch.zeros_like(depth))
        # TODO: consider changing to max(x,0) and ignore 0 pixels in loss. - DONE!
        return depth
        
        

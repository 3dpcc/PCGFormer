import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import math
from util import index_points
import pytorch3d.ops

class Pct(nn.Module):
    def __init__(self, input_channel):
        super(Pct, self).__init__()
        self.transformer=knn_Transformer(channels=input_channel)
        self.transformer_1=knn_Transformer(channels=input_channel)

    def forward(self, x):
        #knn
        x_C=x.C.unsqueeze(0).float()
        x_F=x.F.unsqueeze(0).float()
        dist,idx,_=pytorch3d.ops.knn_points(x_C,x_C,None,None,16)
        new_xyz = index_points(x_C, idx)
        new_feature = index_points(x_F,idx)
        new_feature = new_feature.squeeze(0)
        new_xyz = new_xyz.squeeze(0)
        #knn
        out=self.transformer(x,new_xyz,new_feature)

        # knn
        out_C = out.C.unsqueeze(0).float()
        out_F = out.F.unsqueeze(0).float()
        dist, idx, _ = pytorch3d.ops.knn_points(out_C, out_C, None, None, 16)
        new_xyz = index_points(out_C, idx)
        new_feature = index_points(out_F, idx)
        new_feature = new_feature.squeeze(0)
        new_xyz = new_xyz.squeeze(0)
        # knn
        out = self.transformer_1(out, new_xyz, new_feature)

        return out

class knn_Transformer(nn.Module):
    def __init__(self, channels=8):
        super(knn_Transformer, self).__init__()
        self.attention=SA_Layer(channels)

    def forward(self, out_0,new_xyz,new_feature):
        out_0=self.attention(out_0,new_xyz,new_feature)
        return out_0

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv =torch.nn.Linear(channels,channels)
        self.k_conv = torch.nn.Linear(channels,channels)
        self.v_conv = torch.nn.Linear(channels, channels)
        self.d = math.sqrt(channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.bn = ME.MinkowskiBatchNorm(channels)
        self.conv = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)

    def forward(self, x,new_xyz,new_feature):

        x_q = x.F
        Q=self.q_conv(x_q)
        K=self.k_conv(new_feature)
        K=K.permute(0,2,1)
        attention_map=torch.einsum('ndk,nd->nk', K, Q)

        attention_map=F.softmax(attention_map/self.d,dim=-1)
        # print(attention_map)

        V=self.v_conv(new_feature)
        attention_feature=torch.einsum('nk,nkd->nd', attention_map, V)
        x_att = ME.SparseTensor(features=attention_feature, coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager)
        x_att = self.relu(self.bn((self.conv(x - x_att))))
        x = x + x_att

        return x




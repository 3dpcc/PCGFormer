import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np

import math
import pytorch3d.ops
from util import index_points

# class Encoder(torch.nn.Module):
#   def __init__(self, channels=128):
#     super().__init__()
#     self.conv0 = ME.MinkowskiConvolution(
#         in_channels=3,
#         out_channels=64,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)
#     self.conv0_0 = ME.MinkowskiConvolution(
#         in_channels=64,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)

#     self.conv1 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)
#     self.conv1_0 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)

#     self.conv2 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)
#     self.conv2_0 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)

#     self.relu = ME.MinkowskiReLU(inplace=True)

#   def forward(self, x):
#     out = self.relu(self.conv0_0(self.conv0(x)))
#     out = self.relu(self.conv1_0(self.conv1(out)))
#     out = self.conv2_0(self.conv2(out))

#     return out


# class Decoder(torch.nn.Module):
#   def __init__(self, channels=128):
#     super().__init__()
#     self.deconv0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)
#     self.deconv0_0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)

#     self.deconv1 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)
#     self.deconv1_0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)

#     self.deconv2 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=64,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)
#     self.deconv2_0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=64,
#         out_channels=3,
#         kernel_size= 3,
#         stride=1,
#         bias=True,
#         dimension=3)

#     self.relu = ME.MinkowskiReLU(inplace=True)

#   def forward(self, x):
#     out = self.relu(self.deconv0_0(self.deconv0(x)))
#     out = self.relu(self.deconv1_0(self.deconv1(out)))
#     out = self.deconv2_0(self.deconv2(out))

#     return out

class MLP_block(torch.nn.Module):
    def __init__(self, channels=192):
        super().__init__()
        self.linear0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.linear1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.linear2 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.linear3 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=9,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self,x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(self.relu(self.linear0(x)))))))




class Transformer_block(torch.nn.Module):
    def __init__(self, channels, head, k):
        super(Transformer_block, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.linear = torch.nn.Linear(channels, channels)
        self.layer_norm_2 = nn.LayerNorm(channels)

        self.sa = SA_Layer(channels, head, k)

    def forward(self, x, knn_feature, knn_xyz):
        x1 = x + self.sa(x, knn_feature, knn_xyz)
        x1_F = x1.F

        x1_F = self.layer_norm_1(x1_F)
        x1_F = x1_F + self.linear(x1_F)
        x1_F = self.layer_norm_2(x1_F)

        x1 = ME.SparseTensor(features=x1_F, coordinate_map_key=x1.coordinate_map_key,
                             coordinate_manager=x1.coordinate_manager)

        return x1


class Point_Transformer_Last(torch.nn.Module):
    def __init__(self, block=2, channels=128, head=1, k=16):
        super(Point_Transformer_Last, self).__init__()
        self.head = head
        self.k = k
        self.layers = torch.nn.ModuleList()
        for i in range(block):
            self.layers.append(Transformer_block(channels, head, k))

    def forward(self, x):
        out = x
        x_C = out.C.unsqueeze(0).float()
        # ??p?k???????
        dist, idx, _ = pytorch3d.ops.knn_points(x_C, x_C, K=self.k)
        knn_xyz =  pytorch3d.ops.knn_gather(x_C[:,:,1:], idx)
        center_xyz = x_C[:, :, 1:].unsqueeze(2)

        knn_xyz_norm = knn_xyz - center_xyz
        knn_xyz_norm = knn_xyz_norm.squeeze(0)
        knn_xyz_norm = knn_xyz_norm / knn_xyz_norm.max()

        for transformer in self.layers:
            out_F = out.F.unsqueeze(0).float()
            knn_feature = pytorch3d.ops.knn_gather(out_F[:,:,:], idx).squeeze(0)
            out = transformer(x, knn_feature, knn_xyz_norm)

        return out


class SA_Layer(nn.Module):
    def __init__(self, channels, head=1, k=16):
        super(SA_Layer, self).__init__()
        self.channels = channels
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels + 3, channels)
        self.v_conv = torch.nn.Linear(channels + 3, channels)
        self.d = math.sqrt(channels)
        self.head = head
        self.k = k

    def forward(self, x, knn_feature, knn_xyz):
        x_q = x.F

        new_knn_feature = torch.cat((knn_feature, knn_xyz), dim=2)

        Q = self.q_conv(x_q).view(-1, self.head, self.channels // self.head)
        K = self.k_conv(new_knn_feature).view(-1, self.head, self.k, self.channels // self.head)
        attention_map = torch.einsum('nhd,nhkd->nhk', Q, K)
        attention_map = F.softmax(attention_map / self.d, dim=-1)
        print(attention_map)

        V = self.v_conv(new_knn_feature).view(-1, self.head, self.k, self.channels // self.head)
        attention_feature = torch.einsum('nhk,nhkd->nhd', attention_map, V)
        attention_feature = attention_feature.view(-1, self.channels)

        new_x = ME.SparseTensor(features=attention_feature, coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager)

        return new_x


class ResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


def make_layer(block, block_layers, channels):
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)


class InceptionResNet(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_2 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0_0(x))
        out0 = self.relu(self.conv0_2(self.relu(self.conv0_1(out))))
        out1 = self.relu(self.conv1_1(self.relu(self.conv1_0(x))))
        out = ME.cat(out0, out1)
        return out + x


class Encoder(torch.nn.Module):
    def __init__(self, channels=128, is_deep=False):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        if is_deep: self.block0 = make_layer(
            block=ResNet,
            block_layers=3,
            channels=channels)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        if is_deep: self.block1 = make_layer(
            block=ResNet,
            block_layers=3,
            channels=channels)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0_0(self.conv0(x)))
        if 'block0' in self._modules: out = self.block0(out)
        out = self.relu(self.conv1_0(self.conv1(out)))
        if 'block1' in self._modules: out = self.block1(out)
        out = self.conv2_0(self.conv2(out))

        return out


class Decoder(torch.nn.Module):
    def __init__(self, channels=128, is_deep=False):
        super().__init__()
        self.deconv0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv0_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        if is_deep: self.block0 = make_layer(
            block=ResNet,
            block_layers=3,
            channels=channels)

        self.deconv1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv1_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        if is_deep: self.block1 = make_layer(
            block=ResNet,
            block_layers=3,
            channels=channels)

        self.deconv2 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv2_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.deconv0_0(self.deconv0(x)))
        if 'block0' in self._modules: out = self.block0(out)
        out = self.relu(self.deconv1_0(self.deconv1(out)))
        if 'block1' in self._modules: out = self.block1(out)
        out = self.deconv2_0(self.deconv2(out))

        return out


class HyperEncoder(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv_in = ME.MinkowskiConvolution(
            in_channels=8,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=8,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv_in(x))
        out = self.relu(self.conv0_0(self.conv0(out)))
        out = self.conv1_0(self.conv1(out))
        return out


class HyperDecoder(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.deconv0 = ME.MinkowskiConvolutionTranspose(
            in_channels=8,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv0_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.deconv1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv1_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels * 2,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.deconv_out = ME.MinkowskiConvolutionTranspose(
            in_channels=channels * 2,
            out_channels=8 * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.deconv0_0(self.deconv0(x)))
        out = self.relu(self.deconv1_0(self.deconv1(out)))
        out = self.deconv_out(out)

        return out


###########################################################################
class MaskSparseCNN(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=-1, stride=1,
                 dilation=1, bias=False, dimension=None):
        super(MaskSparseCNN, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            bias=bias,
                                            dimension=dimension)
        n_kernel, _, _ = self.kernel.size()
        mask = torch.zeros(self.kernel.size())
        mask[:n_kernel // 2, :, :] = 1
        mask[n_kernel // 2:, :, :] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.kernel.data *= self.mask

        return super(MaskSparseCNN, self).forward(x)


class ContextModelBase(torch.nn.Module):
    def __init__(self, channels=128):
        super(ContextModelBase, self).__init__()
        self.channels = channels
        self.maskedconv = MaskSparseCNN(in_channels=channels,
                                        out_channels=channels * 2,
                                        kernel_size=5,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv0 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        context = self.maskedconv(x)
        out = self.relu(self.conv0(context))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)
        params = out.F
        loc = params[:, :self.channels]
        scale = params[:, self.channels:]

        return loc, scale.abs()


class ContextModelHyper(torch.nn.Module):
    def __init__(self, channels=128):
        super(ContextModelHyper, self).__init__()
        self.channels = channels
        self.maskedconv = MaskSparseCNN(in_channels=channels,
                                        out_channels=channels * 2,
                                        kernel_size=5,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv0 = ME.MinkowskiConvolution(in_channels=channels * 4,
                                             out_channels=channels * 3,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels * 3,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, hyper):
        context = self.maskedconv(x)
        if context.coordinate_manager == hyper.coordinate_manager:
            context_hyper = ME.cat(context, hyper)
        else:
            context_hyper = ME.SparseTensor(
                features=torch.cat((context.F, hyper.F), dim=-1),
                coordinate_map_key=context.coordinate_map_key,
                coordinate_manager=context.coordinate_manager,
                device=context.device)
        out = self.relu(self.conv0(context_hyper))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)
        params = out.F
        loc = params[:, :self.channels]
        scale = params[:, self.channels:]

        return loc, scale.abs()


class Enhancer(torch.nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block0 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res0 = InceptionResNet(channels=64)
        # self.knn0 = Point_Transformer_Last(block=4, channels=64, head=1, k=16)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block1 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res1 = InceptionResNet(channels=128)
        # self.knn1 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block2 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res2 = InceptionResNet(channels=128)
        # self.knn2 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3
        )
        # self.block3 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res3 = InceptionResNet(channels=64)
        # self.knn3 = Point_Transformer_Last(channels=128, head=1, k=16)

        # self.conv4 = ME.MinkowskiConvolution(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     bias=True,
        #     dimension=3
        # )
        # self.block3 = make_layer(block=ResNet, block_layers=3, channels=32)
        # self.res4 = InsertResNet(channels=64)

        self.conv_out0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv_out1 = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.res0(self.relu(self.conv0(x)))
        # out = self.knn0(out)

        out = self.res1(self.relu(self.conv1(out)))
        # out = self.knn1(out)

        out = self.res2(self.relu(self.conv2(out)))
        # out = self.knn2(out)

        out = self.res3(self.relu(self.conv3(out)))

        # out = self.res4(self.relu(self.conv4(out)))

        out = self.conv_out0(out)
        out = out + self.conv_out1(x)

        return out


class Mutiscale_enhancer(torch.nn.Module):
    def __init__(self, channels = 128):
        super().__init__()
        self.enhancer0 = Enhancer(channels=3)

        self.down1 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.enhancer1 = Enhancer(channels=32)

        self.upsamp1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )

        self.MlP = MLP_block(channels=128)
        self.conv_outx = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=9,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self, x):
        out0 = self.enhancer0(x)

        out1 = self.upsamp1(self.enhancer1(self.down1(x)))

        out = ME.cat(out0, out1)

        return self.MlP(out) + self.conv_outx(x)

# class Enhancer(torch.nn.Module):


#     def __init__(self, channels=128):
#         super().__init__()
#         self.conv0 = ME.MinkowskiConvolution(
#             in_channels=3,
#             out_channels=channels,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.block0 = make_layer(
#             block=ResNet,
#             block_layers=3,
#             channels=channels)
#         self.conv1 = ME.MinkowskiConvolution(
#             in_channels=channels,
#             out_channels=9,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.conv2 = ME.MinkowskiConvolution(
#             in_channels=3,
#             out_channels=9,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.relu = ME.MinkowskiReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.relu(self.conv0(x))
#         out = self.block0(out)
#         out = self.conv1(out)
#         out = out + self.conv2(x)
#
#         return out


class Upsampling_attribute_coords(torch.nn.Module):
    def __init__(self, channels=[1,16,32,64], kenel_size=[5, 3], up=8):
        super().__init__()
        #coords features extract
        self.coords_conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res0 = InceptionResNet(channels=channels[1], kernel_size=kenel_size[0])
        self.coords_conv1 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res1 = InceptionResNet(channels=channels[2], kernel_size=kenel_size[0])
        self.coords_conv2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res2 = InceptionResNet(channels=channels[3], kernel_size=kenel_size[0])
        self.coords_conv3 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3] * 2,
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)

        #attributes features extract
        self.attr_conv0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=channels[1],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.attr_res0 = InceptionResNet(channels=channels[1], kernel_size=kenel_size[0])
        self.attr_conv1 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.attr_res1 = InceptionResNet(channels=channels[2], kernel_size=kenel_size[0])
        self.attr_conv2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.attr_res2 = InceptionResNet(channels=channels[3], kernel_size=kenel_size[0])
        self.attr_conv3 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3] * 2,
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)

        #coords features, attribute features fusion
        self.fusion0 = ME.MinkowskiConvolution(
            in_channels=channels[3] * 4,
            out_channels=channels[3] * 2,
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)

        self.fusion1 = ME.MinkowskiConvolution(
            in_channels=channels[3] * 2,
            out_channels=channels[3] * 2,
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)

        #coords upsampling
        self.coords_up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=up,
            stride=1,
            bias=True,
            dimension=3)
        self.coords_convout = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res3 = InceptionResNet(channels=channels[2], kernel_size=kenel_size[1])
        self.coords_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)

        #attributes upsampling
        self.attr_up_convout = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)

        self.attr_target = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)

        self.conv_out = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=3,
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)


        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned


    def forward(self, attr_sparse, coords_sparse, gt, training):
        #coords
        coords_out = self.relu(self.coords_res0(self.coords_conv0(coords_sparse)))
        coords_out = self.relu(self.coords_res1(self.coords_conv1(coords_out)))
        coords_out = self.relu(self.coords_res2(self.coords_conv2(coords_out)))
        coords_out = self.coords_conv3(coords_out)

        #attribute
        attr_out = self.relu(self.attr_res0(self.attr_conv0(attr_sparse)))
        attr_out = self.relu(self.attr_res1(self.attr_conv1(attr_out)))
        attr_out = self.relu(self.attr_res2(self.attr_conv2(attr_out)))
        attr_out = self.attr_conv3(attr_out)

        #fusion
        fusion_out = ME.SparseTensor(features=torch.cat((coords_out.F, attr_out.F), dim=1), coordinate_manager=coords_out.coordinate_manager,
                                     coordinate_map_key=coords_out.coordinate_map_key)
        fusion_out = self.fusion1(self.relu(self.fusion0(fusion_out)))

        coords_out = ME.SparseTensor(features=fusion_out.F[:,:fusion_out.F.shape[1] // 2], coordinate_manager=coords_out.coordinate_manager,
                                     coordinate_map_key=coords_out.coordinate_map_key)

        attr_out = ME.SparseTensor(features=fusion_out.F[:,fusion_out.F.shape[1] // 2:], coordinate_manager=attr_out.coordinate_manager,
                                     coordinate_map_key=attr_out.coordinate_map_key)

        #coords upsampling
        coords_new = self.relu(self.coords_convout(self.coords_up(coords_out)))
        coords_new = self.coords_res3(coords_new)
        coords_cls = self.coords_cls(coords_new)
        coords_out = self.prune_voxel(coords_new, coords_cls,
                [gt.C.shape[0]], gt, training)
        #attribute features project coords_new

        out = self.relu(self.attr_up_convout(self.attr_target(attr_out, coords_out.C)))

        out = self.conv_out(out)

        return out, coords_cls


class Upsampling_coords(torch.nn.Module):
    def __init__(self, channels=[1, 16, 32, 64], kernel_size=[5, 3], up=8):
        super().__init__()
        # coords features extract
        self.coords_conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kernel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res0 = InceptionResNet(channels=channels[1], kernel_size=kernel_size[0])
        self.coords_conv1 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kernel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res1 = InceptionResNet(channels=channels[2], kernel_size=kernel_size[0])
        self.coords_conv2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=kernel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res2 = InceptionResNet(channels=channels[3], kernel_size=kernel_size[0])
        self.coords_conv3 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3] * 2,
            kernel_size=kernel_size[0],
            stride=1,
            bias=True,
            dimension=3)

        # coords upsampling
        self.coords_up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[3] * 2,
            out_channels=channels[3],
            kernel_size=up,
            stride=1,
            bias=True,
            dimension=3)
        self.coords_convout = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=kernel_size[1],
            stride=1,
            bias=True,
            dimension=3)
        self.coords_res3 = InceptionResNet(channels=channels[2], kernel_size=kernel_size[1])
        self.coords_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=kernel_size[1],
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()


    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned


    def forward(self, coords_sparse, gt, training):
        # coords
        coords_out = self.relu(self.coords_res0(self.coords_conv0(coords_sparse)))
        coords_out = self.relu(self.coords_res1(self.coords_conv1(coords_out)))
        coords_out = self.relu(self.coords_res2(self.coords_conv2(coords_out)))
        coords_out = self.coords_conv3(coords_out)

        # coords upsampling
        coords_new = self.relu(self.coords_convout(self.coords_up(coords_out)))
        coords_new = self.coords_res3(coords_new)
        print(coords_new.F)
        coords_cls = self.coords_cls(coords_new)

        coords_out = self.prune_voxel(coords_new, coords_cls,
                                      [gt.C.shape[0]], gt, training)
        # attribute features project coords_new

        return coords_out, coords_cls


class Upsampling_attr(torch.nn.Module):
    def __init__(self, channels=[1, 16, 32, 64], kenel_size=[5, 3], up=8):
        super().__init__()
        #attributes features extract
        self.attr_conv0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=channels[1],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.attr_res0 = InceptionResNet(channels=channels[1], kernel_size=kenel_size[0])
        self.attr_conv1 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.attr_res1 = InceptionResNet(channels=channels[2], kernel_size=kenel_size[0])
        self.attr_conv2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)
        self.attr_res2 = InceptionResNet(channels=channels[3], kernel_size=kenel_size[0])
        self.attr_conv3 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3] * 2,
            kernel_size=kenel_size[0],
            stride=1,
            bias=True,
            dimension=3)

        # attr target
        #attributes upsampling
        self.attr_up_convout = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)

        self.attr_target = ME.MinkowskiConvolution(
            in_channels=channels[3] * 2,
            out_channels=channels[3],
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)

        self.conv_out = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=3,
            kernel_size=kenel_size[1],
            stride=1,
            bias=True,
            dimension=3)


        self.relu = ME.MinkowskiReLU(inplace=True)


    def forward(self, attr_sparse, coords_sparse, gt, training):
        #attribute
        attr_out = self.relu(self.attr_res0(self.attr_conv0(attr_sparse)))
        attr_out = self.relu(self.attr_res1(self.attr_conv1(attr_out)))
        attr_out = self.relu(self.attr_res2(self.attr_conv2(attr_out)))
        attr_out = self.attr_conv3(attr_out)
        #attribute features project coords_new
        out = self.relu(self.attr_up_convout(self.attr_target(attr_out, coords_sparse.C)))

        out = self.conv_out(out)

        return out
if __name__ == '__main__':
    # encoder = Encoder(128, 3)
    # print(encoder)
    # decoder = Decoder(128, 3)
    # print(decoder)
    #
    # hyperEncoder = HyperEncoder(128)
    # print(hyperEncoder)
    # hyperDecoder = HyperDecoder(128)
    # print(hyperDecoder)
    #
    # contextModelBase = ContextModelBase(128)
    # print(contextModelBase)
    #
    # contextModelHyper = ContextModelHyper(128)
    # print(contextModelHyper)
    enhance = Mutiscale_enhancer()
    print(enhance)
    print('params:', sum(param.numel() for param in enhance.parameters()))

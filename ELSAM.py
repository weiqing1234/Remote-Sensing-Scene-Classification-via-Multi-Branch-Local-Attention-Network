import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)    # 将前面多维度的tensor展平成一维

# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 print('avg_pool.shape',avg_pool.shape)
#                 channel_att_raw = self.mlp( avg_pool )
#                 print('channel_att_raw.shape',channel_att_raw.shape)
#             elif pool_type=='max':
#                 max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 print('max_pool.shape',max_pool.shape)
#                 channel_att_raw = self.mlp( max_pool )
#                 print('channel_att_raw.shape',channel_att_raw.shape)
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#                 print('channel_att_sum.shape',channel_att_sum.shape)
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw
#                 print('channel_att_sum.shape',channel_att_sum.shape)

#         scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
#         print('scale.shape',scale.shape)
#         return x * scale

# class ChannelGate(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelGate, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         print(self.avg_pool(x).shape)
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)


class Enhanced_Channel_Attenion(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(Enhanced_Channel_Attenion, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(2)
        # self.ma_pool = nn.AdaptiveAvgPool2d(4)
        self.down_op = nn.Conv2d(1, 1, kernel_size=(2, 1))
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        # self.sharedMLP = nn.Sequential(
        #     nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, bias=False), nn.ReLU(),
        #     nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, bias=False))


    def forward(self, x):

        r=x
        b, c, _, _=x.size()
        # x_avg = self.avg_pool(x).view(b, c)
        # x_max = self.max_pool(x).view(b, 4 * c)
        # x_ma= self.ma_pool(x).view(b, 16 * c)
        # print(self.avg_pool(x).view(b, c).shape)
        # print(self.max_pool(x).view(b, 4*c).shape)
        # print(torch.cat((x_avg, x_max, x_ma), 1).shape)

        x_avg = self.avg_pool(x).view(self.avg_pool(x).size(0), -1).unsqueeze(axis=1).unsqueeze(axis=1)
        # # print(x_avg.shape)
        x_max = self.max_pool(x).view(self.max_pool(x).size(0), -1).unsqueeze(axis=1).unsqueeze(axis=1)
        # x_avg = self.avg_pool(x).permute(0, 2, 3, 1)
        # x_max = self.max_pool(x).permute(0, 2, 3, 1)
        # print(x_avg.shape, x_max.shape)
        x = torch.cat((x_avg, x_max), dim=2)
        # print(x.shape)
        x = self.down_op(x)
        # print(x.shape)
        # x = F.flatten(x)
        # x = self.sharedMLP(x)
        x = self.mlp(x)
        # print(x.shape)
        x=torch.sigmoid(x)
        # print(x.shape)
        # x=F.elemwise_add(x,F.ones_like(x))
        # x=x.unsqueeze(2).unsqueeze(3).expand_as(r)
        # print(x.shape)
        # print('x*r',(x*r).shape)
        # x = x * r
        return x


class ChannelPool(nn.Module):
    def forward(self, x, k):   # k is stride of local maxpool and avgpool
        avg_pool = F.avg_pool2d(x, k, stride=k)
        max_pool = F.max_pool2d(x, k, stride=k)
        return torch.cat( (torch.max(max_pool,1)[0].unsqueeze(1), torch.mean(avg_pool,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=3, dilation=3, relu=False)
        #self.sample = nn.UpsamplingNearest2d(scale_factor=4)

    def forward(self, x):
        # x_compress = self.compress(x)
        # x_out = self.sample(self.spatial(x_compress))
        r = x
        b, c, h, w = x.size()
        # x1 = self.compress(x, 1).view(b, self.compress(x, 1).size(1), -1) # 8 is stride of local maxpool and avgpool
        # x2 = self.compress(x, 2).view(b, self.compress(x, 2).size(1), -1)
        # x3 = self.compress(x, 4).view(b, self.compress(x, 4).size(1), -1)

        # x2 = x2.view(b, x2.size(1), 4 * x2.size(2))
        # print('x_compress',x1.shape, x2.shape, x3.shape)

        # x = torch.cat((x1,x2,x3), dim=1)
        # print(x.shape)

        x_compress = self.compress(x, 2)
        x_out = self.spatial(x_compress)
        # print('x_out',x_out.shape)
        x_out = F.interpolate(x_out, size=[h, w], mode='nearest')
        # # print('x_out',x_out.shape)
        x = torch.sigmoid(x_out) # broadcasting
        # print('scale',scale.shape)
        # print('x*scale',(x * scale).shape)
        # x = x * r
        return x


class ELSAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(ELSAM, self).__init__()
        self.ChannelGate = Enhanced_Channel_Attenion(gate_channels, reduction_ratio)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        # x_out1 = self.ChannelGate(x)
        x_out1 = self.ChannelGate(x).unsqueeze(axis=2).unsqueeze(axis=2)

        if not self.no_spatial:
            x_out2 = self.SpatialGate(x)

        # x_out = torch.add(x_out1, x_out2)
        x_out = torch.mul(x_out1, x_out2)
        x_out = torch.sigmoid(x_out)
        # print(x_out.shape)

        return x_out

def main():
    x = torch.rand([16, 256, 56, 56])
    mode = ELSAM(256)
    y = mode(x)
    print(get_n_params(mode))

if __name__ == '__main__':
    main()
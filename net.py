import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from t2t_vit import Channel, Spatial
from function import adaptive_instance_normalization
import layers as L
from loss import add_edges_to_image
from args_fusion import args


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out


class ConvLayerwithattention(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayerwithattention, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = L.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayerwithattention(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels)
        self.conv2 = ConvLayerwithattention(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels // 2, kernel_size, stride)
        self.conv2 = ConvLayer(in_channels // 2, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class CAF(nn.Module):
    def __init__(self):
        super(CAF, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)

        self.convfuse = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnfuse = nn.BatchNorm2d(64)

        self.conv1f = nn.Conv2d(192, 192, kernel_size=1)
        self.safused = sa_layer(192)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=True)
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        # mul = out1h * out1v
        mul = torch.mul(out1h, out1v)
        fuse = torch.cat((out1h, out1v), 1)
        fuse = torch.cat((fuse, mul), 1)

        # gap = nn.AdaptiveAvgPool2d((1, 1))(fuse)
        # out1f = nn.Softmax(dim=1)(self.conv1f(gap)) * gap.shape[1]
        # out1f = nn.Sigmoid()(self.conv1f(gap))
        out1f = self.safused(fuse)
        # fuse_channel = out1f * fuse
        fuse_channel = torch.mul(out1f, fuse)

        out3h = F.relu(self.bn3h(self.conv3h(fuse_channel)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse_channel)), inplace=True)

        out4h = F.relu(self.bn4h(self.conv4h(out3h + out1h)), inplace=True)
        out4v = F.relu(self.bn4v(self.conv4v(out3v + out1v)), inplace=True)

        out = torch.cat((out4h, out4v), 1)
        out = F.relu(self.bnfuse(self.convfuse(out)), inplace=True)
        return out




# 1
class net(nn.Module):
    def __init__(self, input_nc=2, output_nc=1):
        super(net, self).__init__()
        kernel_size = 1
        stride = 1

        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(4)
        self.down3 = nn.AvgPool2d(8)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_in1 = ConvLayer(input_nc, input_nc, kernel_size, stride)
        self.conv_out = ConvLayer(64, 1, kernel_size, stride, is_last=True)
        # self.conv_t3 = ConvLayer(128, 64, kernel_size=1, stride=1)
        # self.conv_t2 = ConvLayer(64, 32, kernel_size=1, stride=1)
        # self.conv_t0 = ConvLayer(3, 3, kernel_size, stride)

        self.en0 = Encoder(2, 64, kernel_size, stride)
        self.en1 = Encoder(64, 64, kernel_size, stride)
        self.en2 = Encoder(64, 64, kernel_size, stride)
        self.en3 = Encoder(64, 64, kernel_size, stride)

        # self.de3 = Decoder(96, 32, kernel_size, stride)
        # self.de2 = Decoder(48, 16, kernel_size, stride)
        # self.de1 = Decoder(19, 3, kernel_size, stride)
        # self.de0 = Decoder(3, 3, kernel_size, stride)

        # self.f1 = ConvLayer(6, 3, kernel_size, stride)
        # self.f2 = ConvLayer(32, 16, kernel_size, stride)
        # self.f3 = ConvLayer(64, 32, kernel_size, stride)

        # self.ctrans0 = Channel(size=256, embed_dim=128, patch_size=16, channel=3)
        # self.ctrans1 = Channel(size=128, embed_dim=128, patch_size=16, channel=16)
        # self.ctrans2 = Channel(size=64, embed_dim=128, patch_size=16, channel=32)
        self.ctrans3 = Channel(size=32, embed_dim=128, patch_size=16, channel=64)

        # self.strans0 = Spatial(size=256, embed_dim=128*2, patch_size=8, channel=3)
        # self.strans1 = Spatial(size=128, embed_dim=256*2, patch_size=8, channel=16)
        # self.strans2 = Spatial(size=256, embed_dim=512*2, patch_size=8, channel=32)
        self.strans3 = Spatial(size=256, embed_dim=1024 * 2, patch_size=4, channel=64)

        self.conv_up1 = ConvLayerwithattention(64, 64, kernel_size, stride)
        self.conv_up2 = ConvLayerwithattention(64, 64, kernel_size, stride)
        self.conv_up3 = ConvLayerwithattention(64, 64, kernel_size, stride)
        self.edgefusion = CAF()
        self.sa_attention =sa_layer(64)

    # def en(self, vi, ir):
    #     f = torch.cat([vi, ir], dim=1)
    #     x = self.conv_in1(f)
    #     x0 = self.en0(x)
    #     # x1 = self.en1(self.down1(x0))
    #     # x2 = self.en2(self.down1(x1))
    #     # x3 = self.en3(self.down1(x2))
    #     x1 = self.en1(x0)
    #     x2 = self.en2(x1)
    #     x3 = self.en3(x2)
    #
    #     return [x0, x1, x2, x3]

    # def de(self, f):
    #     x0, x1, x2, x3 = f
    #     o3 = self.de3(torch.cat([self.up1(x3), x2], dim=1))
    #     o2 = self.de2(torch.cat([self.up1(o3), x1], dim=1))
    #     o1 = self.de1(torch.cat([self.up1(o2), x0], dim=1))
    #     o0 = self.de0(o1)
    #     out = self.conv_out1(o0)
    #     return out

    def forward(self, vi, ir):
        vi = vi.cpu()
        ir = ir.cpu()
        _, vi_edge = add_edges_to_image(vi)
        _, ir_edge = add_edges_to_image(ir)
        if args.cuda:
            vi = vi.cuda()
            ir = ir.cuda()
            vi_edge = vi_edge.cuda()
            ir_edge = ir_edge.cuda()
        # w = ir / (torch.max(ir) - torch.min(ir))
        # f_pre = w * ir + (1-w) * vi
        f0 = torch.cat([vi, ir], dim=1)
        f0_edge = torch.cat([vi_edge, ir_edge], dim=1)
        x = self.conv_in1(f0)
        x_edge = self.conv_in1(f0_edge)
        x0 = self.en0(x)
        x0_edge = self.en0(x_edge)
        x0_edge_attention_maps = self.down1(self.sa_attention(x0_edge))
        x1 = self.en1(self.down1(x0))
        x1_edge = self.en1(self.down1(x0_edge))
        x1_edge_attention_maps = self.down1(self.sa_attention((x1_edge * x0_edge_attention_maps)+x1_edge))
        x2 = self.en2(self.down1(x1))
        x2_edge = self.en2(self.down1(x1_edge))
        x2_edge_attention_maps = self.down1(self.sa_attention(((x2_edge * x1_edge_attention_maps) + x2_edge)))
        x3 = self.en3(self.down1(x2))
        x3_edge = self.en3(self.down1(x2_edge))
        x3_edge = (x3_edge * x2_edge_attention_maps) + x3_edge
        # x1 = self.en1(x0)
        # x2 = self.en2(x1)
        # x3 = self.en3(x2)
        x3t =self.edgefusion(x3,x3_edge)

        x3t = self.strans3(self.ctrans3(x3t))
        # x3t = x3 + x3_edge
        # x2r = self.ctrans2(x2)
        # x1r = self.ctrans1(x1)
        # x0r = self.ctrans0(x0)
        # x3m = torch.clamp(x3r, 0, 1)
        #         没有conv+attention
        # x3m = x3t
        # x3r = x3 * x3m
        # x2m = self.up1(x3m)
        # x2r = x2 * x2m
        # x1m = self.up1(x2m) + self.up2(x3m)
        # x1r = x1 * x1m
        # x0m = self.up1(x1m) + self.up2(x2m) + self.up3(x3m)
        # x0r = x0 * x0m
        #         有conv+attention
        x3m = x3t
        x3r = x3 * x3m
        x2m = self.up1(x3m)
        x2r = self.conv_up1(x2 * x2m)
        x1m = self.up1(x2m) + self.up2(x3m)
        x1r = self.conv_up2(x1 * x1m)
        x0m = self.up1(x1m) + self.up2(x2m) + self.up3(x3m)
        x0r = self.conv_up3(x0 * x0m)

        other = self.up3(x3r) + self.up2(x2r) + self.up1(x1r) + x0r
        f1 = self.conv_out(other)
        # out = self.conv_out(f1)

        return f1




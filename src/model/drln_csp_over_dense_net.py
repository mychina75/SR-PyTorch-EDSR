import torch
import torch.nn as nn
import model.ops as ops
import torch.nn.functional as F


def make_model(args, parent=False):
    return DRLN(args)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel, channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel, channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel, channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        in_channels = int(in_channels / 2)
        out_channels = int(out_channels / 2)

        self.r1 = ops.ResidualBlock(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ops.ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = ops.BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels*2)

    def forward(self, x):
        c0 = x[:, int(x.shape[1] / 2):, :, :]

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = torch.cat([x[:, :int(x.shape[1] / 2), :, :], g], dim=1)
        out = self.ca(out)

        return out


class DRLN(nn.Module):
    def __init__(self, args):
        super(DRLN, self).__init__()

        # n_resgroups = args.n_resgroups
        # n_resblocks = args.n_resblocks
        # n_feats = args.n_feats
        # kernel_size = 3
        # reduction = args.reduction
        # scale = args.scale[0]
        # act = nn.ReLU(True)

        self.scale = args.scale[0]
        chs = 64

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.head = nn.Conv2d(3, chs, 3, 1, 1)

        self.b1 = Block(chs//2, chs//2)
        self.b2 = Block(chs//2, chs//2)
        self.b3 = Block(chs//2, chs//2)
        self.b4 = Block(chs//2, chs//2)
        self.b5 = Block(chs//2, chs//2)
        self.b6 = Block(chs//2, chs//2)
        self.b7 = Block(chs//2, chs//2)
        self.b8 = Block(chs//2, chs//2)
        self.b9 = Block(chs//2, chs//2)
        self.b10 = Block(chs//2, chs//2)
        self.b11 = Block(chs//2, chs//2)
        self.b12 = Block(chs//2, chs//2)
        self.b13 = Block(chs//2, chs//2)
        self.b14 = Block(chs//2, chs//2)
        self.b15 = Block(chs//2, chs//2)
        self.b16 = Block(chs//2, chs//2)
        self.b17 = Block(chs//2, chs//2)
        self.b18 = Block(chs//2, chs//2)
        self.b19 = Block(chs//2, chs//2)
        self.b20 = Block(chs//2, chs//2)

        self.c1 = ops.BasicBlock(chs//2 * 2, chs//2, 3, 1, 1)
        self.c2 = ops.BasicBlock(chs//2 * 3, chs//2, 3, 1, 1)
        self.c3 = ops.BasicBlock(chs//2 * 4, chs//2, 3, 1, 1)
        self.c4 = ops.BasicBlock(chs//2 * 2, chs//2, 3, 1, 1)
        self.c5 = ops.BasicBlock(chs//2 * 3, chs//2, 3, 1, 1)
        self.c6 = ops.BasicBlock(chs//2 * 4, chs//2, 3, 1, 1)
        self.c7 = ops.BasicBlock(chs//2 * 2, chs//2, 3, 1, 1)
        self.c8 = ops.BasicBlock(chs//2 * 3, chs//2, 3, 1, 1)
        self.c9 = ops.BasicBlock(chs//2 * 4, chs//2, 3, 1, 1)
        self.c10 = ops.BasicBlock(chs//2 * 2, chs//2, 3, 1, 1)
        self.c11 = ops.BasicBlock(chs//2 * 3, chs//2, 3, 1, 1)
        self.c12 = ops.BasicBlock(chs//2 * 4, chs//2, 3, 1, 1)
        self.c13 = ops.BasicBlock(chs//2 * 2, chs//2, 3, 1, 1)
        self.c14 = ops.BasicBlock(chs//2 * 3, chs//2, 3, 1, 1)
        self.c15 = ops.BasicBlock(chs//2 * 4, chs//2, 3, 1, 1)
        self.c16 = ops.BasicBlock(chs//2 * 5, chs//2, 3, 1, 1)
        self.c17 = ops.BasicBlock(chs//2 * 2, chs//2, 3, 1, 1)
        self.c18 = ops.BasicBlock(chs//2 * 3, chs//2, 3, 1, 1)
        self.c19 = ops.BasicBlock(chs//2 * 4, chs//2, 3, 1, 1)
        self.c20 = ops.BasicBlock(chs//2 * 5, chs//2, 3, 1, 1)

        self.c21 = ops.BasicBlock(chs, chs, 3, 1, 1)
        self.c22 = ops.BasicBlock(chs, chs, 3, 1, 1)
        self.c23 = ops.BasicBlock(chs, chs, 3, 1, 1)
        self.c24 = ops.BasicBlock(chs, chs, 3, 1, 1)
        self.c25 = ops.BasicBlock(chs, chs, 3, 1, 1)
        self.c26 = ops.BasicBlock(chs, chs, 3, 1, 1)

        self.upsample = ops.UpsampleBlock(chs, self.scale, multi_scale=False)
        # self.convert = ops.ConvertBlock(chs, chs, 20)
        self.tail = nn.Conv2d(chs, 3, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        fh = x[:, :int(x.shape[1] / 2), :, :] #first half
        c0 = o0 = x[:, int(x.shape[1] / 2):, :, :] #second half

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        o3 = torch.cat([fh, o3], dim=1)
        o3 = self.c21(o3)
        a1 = o3 + x

        fh = a1[:, :int(x.shape[1] / 2), :, :]
        sh = a1[:, int(x.shape[1] / 2):, :, :]
        b4 = self.b4(sh)
        c4 = torch.cat([o3, b4], dim=1)
        o4 = self.c4(c4)

        b5 = self.b5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)

        o6 = torch.cat([fh, o6], dim=1)
        o6 = self.c22(o6)
        a2 = o6 + a1

        fh = a2[:, :int(x.shape[1] / 2), :, :]
        sh = a2[:, int(x.shape[1] / 2):, :, :]
        b7 = self.b7(sh)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)

        o9 = torch.cat([fh, o9], dim=1)
        o9 = self.c23(o9)
        a3 = o9 + a2

        fh = a3[:, :int(x.shape[1] / 2), :, :]
        sh = a3[:, int(x.shape[1] / 2):, :, :]
        b10 = self.b10(sh)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)

        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)

        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)

        o12 = torch.cat([fh, o12], dim=1)
        o12 = self.c24(o12)
        a4 = o12 + a3

        fh = a4[:, :int(x.shape[1] / 2), :, :]
        sh = a4[:, int(x.shape[1] / 2):, :, :]
        b13 = self.b13(sh)
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)

        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)

        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)

        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)

        o16 = torch.cat([fh, o16], dim=1)
        o16 = self.c25(o16)
        a5 = o16 + a4

        fh = a5[:, :int(x.shape[1] / 2), :, :]
        sh = a5[:, int(x.shape[1] / 2):, :, :]
        b17 = self.b17(sh)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)

        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)

        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)

        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)

        o20 = torch.cat([fh, o20], dim=1)
        o20 = self.c26(o20)
        a6 = o20 + a5

        # c_out = torch.cat([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20], dim=1)

        # b = self.convert(c_out)
        b_out = a6 + x
        out = self.upsample(b_out, scale=self.scale)

        out = self.tail(out)
        f_out = self.add_mean(out)

        return f_out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('upsample') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))



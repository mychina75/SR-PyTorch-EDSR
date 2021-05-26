import torch
import torch.nn as nn
import pdb
import model.common
import model.ops as ops

def make_model(args, parent=False):
    return HAN(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0, 1)
        self.c2 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0, 1)
        self.c3 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0, 1)
        self.c4 = ops.BasicBlock((channel // reduction)*3, channel, 1, 1, 0, 1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.c5 = ops.BasicBlock(channel, channel // reduction, 1, 1, 0, 1)
        self.c6 = ops.BasicBlock(channel, channel // reduction, 1, 1, 0, 1)
        self.c7 = ops.BasicBlock(channel, channel // reduction, 1, 1, 0, 1)
        self.c8 = ops.BasicBlock((channel // reduction) * 3, channel, 1, 1, 0, 1)

        self.c9 = ops.BasicBlock(2, 1, 1, 1, 0, 1)
        sigmoid = nn.Sigmoid()
        #self.c9 = ops.BasicBlockSig(channel * 2, channel, 1, 1, 0)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()

        y1 = self.avg_pool(x)
        c1 = self.c1(y1)
        c2 = self.c2(y1)
        c3 = self.c3(y1)
        c_out1 = torch.cat([c1, c2, c3], dim=1)
        y1 = self.c4(c_out1)

        y2 = self.max_pool(x)
        c5 = self.c5(y2)
        c6 = self.c6(y2)
        c7 = self.c7(y2)
        c_out2 = torch.cat([c5, c6, c7], dim=1)
        y2 = self.c8(c_out2)

        y3 = y1.view(m_batchsize, 1, C, 1)
        y4 = y2.view(m_batchsize, 1, C, 1)

        c_out = torch.cat([y3, y4], dim=1)
        c9 = self.c9(c_out)

        y5 = c9.view(m_batchsize, C, 1, 1)
        y = self.sigmoid(y5)

        return x * y

class PALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PALayer, self).__init__()

        self.c1 = ops.BasicBlock(channel, channel // reduction, 1, 1, 0, 1)
        self.c2 = ops.BasicBlock(channel, channel // reduction, 3, 1, 1, 1)
        self.c3 = ops.BasicBlock(channel, channel // reduction, 3, 1, 1, 1)
        self.c4 = ops.BasicBlock(channel, channel // reduction, 3, 1, 1, 1)
        self.c5 = ops.BasicBlockSig((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        # y = self.avg_pool(x)
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c_out = torch.cat([c1, c2, c4], dim=1)
        y = self.c5(c_out)
        return x * y

class LAM_Module(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""

    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Holistic Attention Network (HAN)
class HAN(nn.Module):
    def __init__(self, args, conv=model.common.default_conv):
        super(HAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4542, 0.4609, 0.4557)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = model.common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            model.common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = model.common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats * 11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        # pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            # print(name)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)
        # res = self.body(x)
        out1 = res
        # res3 = res.unsqueeze(1)
        # res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)

        res += x
        # res = self.csa(res)

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
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
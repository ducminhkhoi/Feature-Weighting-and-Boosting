import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import get_indices_of_pairs
from torch.optim import Adam, SGD
from copy import deepcopy


__all__ = ['VGG', 'ResNet', 'vgg16', 'resnet101', ]


class FewShotModel(nn.Module):

    def __init__(self, model, num_shots):
        super().__init__()

        self.model = model 
        self.num_shots = num_shots

    def forward_all(self, x):
        raise NotImplementedError

    def forward(self, x, training=True, step=1):
        xq, mq, xs, ms = x

        if training:
            return self.forward_train(xq, xs, ms)
        else:
            if self.model in [0, 1, 2, 4]:
                return self.forward_val(xq, xs, ms)
            elif self.model == 8:
                return self.forward_val_5shot(xq, xs, ms)
            elif self.model == 9:
                return self.forward_val_ensemble_5shot(xq, mq, xs, ms, step)
            else:
                return self.forward_val_ensemble(xq, mq, xs, ms, step)

    def forward_train(self, xq, xs, ms):
        Fl, _, _, Fq = self.forward_all(xq)

        with torch.no_grad():
            self.eval()

            # early filtered feature
            xs_ = xs.clone()
            for i in range(len(xs_)):
                xs_[i, :, ms[i]==0] = xs[i, :, ms[i]==0].mean(-1, keepdim=True)
                
            Fs_ = self.forward_all(xs_)[-1]
            Fs_ = F.adaptive_max_pool2d(Fs_, 1)

            # late filtered feature
            Fs = self.forward_all(xs)[-1]
            Ms = F.interpolate(ms.unsqueeze(1).float(), Fs.shape[-2:]).squeeze(1)
            c_Fs = torch.stack([fs[:, ms==0].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])
            Fs = torch.stack([fs[:, ms==1].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])

            r = F.normalize(Fs - c_Fs, p=2)
            # r = Fs - c_Fs

            self.train()

        s_ = F.relu(F.cosine_similarity(Fq, Fs_).unsqueeze(1))
        s_ = F.interpolate(s_, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max_ = F.adaptive_max_pool2d(s_, 1)
        s_ = s_ / (s_max_ + 1e-16)

        if self.model in [0, 1, 3, 6]:
            s = F.relu(F.cosine_similarity(Fq, Fs).unsqueeze(1))
        else:
            s = F.relu(F.cosine_similarity(Fq * r, Fs * r).unsqueeze(1))

        s = F.interpolate(s, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max = F.adaptive_max_pool2d(s, 1)
        s = s / (s_max + 1e-16)

        if self.model in [0, 2, 3, 5, 8, 9]:
            x = torch.cat([Fl * s], 1)
        else:
            x = torch.cat([Fl * s, Fl * s_], 1)

        x = self.exit_layer(x)

        x = F.interpolate(x, xq.shape[-2:], mode='bilinear', align_corners=True)

        return s, x, Fs, r

    def forward_val(self, xq, xs, ms):
        Fl, _, _, Fq = self.forward_all(xq)

        # early filtered feature
        xs_ = xs.clone()
        for i in range(len(xs_)):
            xs_[i, :, ms[i]==0] = xs[i, :, ms[i]==0].mean(-1, keepdim=True)
            
        Fs_ = self.forward_all(xs_)[-1]
        Fs_ = F.adaptive_max_pool2d(Fs_, 1)

        # late filtered feature
        Fs = self.forward_all(xs)[-1]
        Ms = F.interpolate(ms.unsqueeze(1).float(), Fs.shape[-2:]).squeeze(1)
        c_Fs = torch.stack([fs[:, ms==0].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])
        Fs = torch.stack([fs[:, ms==1].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])

        r = F.normalize(Fs - c_Fs, p=2)

        s_ = F.relu(F.cosine_similarity(Fq, Fs_).unsqueeze(1))
        s_ = F.interpolate(s_, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max_ = F.adaptive_max_pool2d(s_, 1)
        s_ = s_ / (s_max_ + 1e-16)
        
        if self.model in [0, 1, 3, 6]:
            s = F.relu(F.cosine_similarity(Fq, Fs).unsqueeze(1))
        else:
            s = F.relu(F.cosine_similarity(Fq * r, Fs * r).unsqueeze(1))

        s = F.interpolate(s, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max = F.adaptive_max_pool2d(s, 1)
        s = s / (s_max + 1e-16)

        if self.model in [0, 2, 3, 5, 8, 9]:
            x = torch.cat([Fl * s], 1)
        else:
            x = torch.cat([Fl * s, Fl * s_], 1)

        x = self.exit_layer(x)

        x = F.interpolate(x, xq.shape[-2:], mode='bilinear', align_corners=True)

        return s, x, Fs, Fs_

    def forward_val_5shot(self, xq, xs, ms):
        Fl, _, _, Fq = self.forward_all(xq)

        Fs = [self.forward_all(x)[-1][0] for x in xs]
        Ms = [F.interpolate(m.unsqueeze(1).float(), f.shape[-2:]).squeeze(1).squeeze(0) for m, f in zip(ms, Fs)]

        c_Fs = torch.stack([fs[:, ms==0].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])
        Fs_ = torch.stack([fs[:, ms==1].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])

        c_Fs = c_Fs.sum(0, keepdim=True)
        Fs = Fs_.sum(0, keepdim=True)

        r = F.normalize(Fs - c_Fs, p=2)

        s = F.relu(F.cosine_similarity(Fq * r, Fs * r).unsqueeze(1))
        s = F.interpolate(s, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max = F.adaptive_max_pool2d(s, 1)
        s = s / (s_max + 1e-16)

        x = torch.cat([Fl * s], 1)

        x = self.exit_layer(x)

        x = F.interpolate(x, xq.shape[-2:], mode='bilinear', align_corners=True)

        return s, x, Fs, None

    def forward_val_ensemble(self, xq, mq, xs, ms, step=10):
        """optimize feature fs"""

        Fl, _, _, Fq = self.forward_all(xq)

        # early filtered feature
        xs_ = xs.clone()
        for i in range(len(xs_)):
            xs_[i, :, ms[i]==0] = xs[i, :, ms[i]==0].mean(-1, keepdim=True)
            
        Fs_ = self.forward_all(xs_)[-1]
        Fs_ = F.adaptive_max_pool2d(Fs_, 1)

        # late filtered feature
        Fl_ori, _, _, Fs = self.forward_all(xs)
        Fs_ori = Fs
        Ms = F.interpolate(ms.unsqueeze(1).float(), Fs.shape[-2:]).squeeze(1)
        c_Fs = torch.stack([fs[:, ms==0].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])
        Fs = torch.stack([fs[:, ms==1].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])

        r = F.normalize(Fs - c_Fs, p=2)

        # working on support image to find the contribution of seen classes to new class
        s_ = F.relu(F.cosine_similarity(Fs_ori, Fs_).unsqueeze(1))
        s_ = F.interpolate(s_, Fl_ori.shape[-2:], mode='bilinear', align_corners=True)
        s_max_ = F.adaptive_max_pool2d(s_, 1)
        s_ = s_ / (s_max_ + 1e-16)
        Fl_ori_ = Fl_ori * s_

        if self.model in [0, 1, 3, 6]:
            fs = Fs
        else:
            fs = Fs * r
            Fs_ori = Fs_ori * r

        list_fs_ = []
        list_iou = []

        list_x = []
        f = nn.Parameter(fs)
        #optimizer = torch.optim.SGD([f], lr=1e2)
        optimizer = torch.optim.Adam([f], lr=1e-2)
        
        # use gradient boosting to find fs_
        for i in range(step+1):
            with torch.enable_grad():
                list_fs_.append(f.clone())

                s = F.relu(F.cosine_similarity(Fs_ori, f)).unsqueeze(1)
                s = F.interpolate(s, Fl_ori.shape[-2:], mode='bilinear', align_corners=True)
                s_max = F.adaptive_max_pool2d(s, 1)
                s = s / (s_max + 1e-16)

                if self.model in [0, 2, 3, 5, 8, 9]:
                    x = torch.cat([Fl_ori * s,], 1)
                else:
                    x = torch.cat([Fl_ori * s, Fl_ori_], 1)

                x = self.exit_layer(x)

                x = F.interpolate(x, xs.shape[-2:], mode='bilinear', align_corners=True)

                loss = F.cross_entropy(x, ms)

                # self.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                x = x.argmax(1)
                list_x.append(x[0])
            
                iou = ((x == 1) & (ms == 1)).sum().float() / ((x == 1) | (ms == 1)).sum().float()
                list_iou.append(iou)   
        
        list_iou = torch.stack(list_iou)

        list_fs_ = torch.cat(list_fs_)

        ###############################################################
        # working on query image and use list_iou as contribution to find the final mask
        s_ = F.relu(F.cosine_similarity(Fq, Fs_).unsqueeze(1))
        s_ = F.interpolate(s_, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max_ = F.adaptive_max_pool2d(s_, 1)
        s_ = s_ / (s_max_ + 1e-16)
        Fl_ = Fl * s_

        if self.model in [0, 1, 3, 6]:
            s = F.relu(F.cosine_similarity((Fq).unsqueeze(1), (list_fs_).unsqueeze(0), dim=2))
        else:
            s = F.relu(F.cosine_similarity((Fq * r).unsqueeze(1), (list_fs_).unsqueeze(0), dim=2))

        s = F.interpolate(s, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max = F.adaptive_max_pool2d(s, 1)
        s = s / (s_max + 1e-16)
        s = s.permute(1, 0, 2, 3)

        Fl = Fl * s
        Fl_ = Fl_.expand_as(Fl)

        if self.model in [0, 2, 3, 5, 8, 9]:
            x = torch.cat([Fl], 1)
        else:
            x = torch.cat([Fl, Fl_], 1)

        x = self.exit_layer(x)

        x = F.interpolate(x, xq.shape[-2:], mode='bilinear', align_corners=True)

        x = x.argmax(1)
        
        list_x = x.float()

        list_iou1 = ((x == 1) & (mq == 1)).sum(-1).sum(-1).float() / ((x == 1) | (mq == 1)).sum(-1).sum(-1).float()

        final_x1 = (list_x * list_iou1[..., None, None]).sum(0, keepdim=True)
        final_x = (list_x * list_iou[..., None, None]).sum(0, keepdim=True)

        return r, final_x, final_x1, Fs_

    def forward_val_ensemble_5shot(self, xq, mq, xs, ms, step=10):
        Fl, _, _, Fq = self.forward_all(xq)

        list_Fs, list_Fl_ori = [], []
        for x in xs:
            Fl_ori, _, _, Fs = self.forward_all(x)
            list_Fl_ori.append(Fl_ori)
            list_Fs.append(Fs)

        Fs = [x[0] for x in list_Fs]
        Fs_ori = list_Fs
        Ms = [F.interpolate(m.unsqueeze(1).float(), f.shape[-2:]).squeeze(1).squeeze(0) for m, f in zip(ms, Fs)]

        c_Fs = torch.stack([fs[:, ms==0].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])
        Fs_ = torch.stack([fs[:, ms==1].mean(-1)[..., None, None] for fs, ms in zip(Fs, Ms)])

        c_Fs = c_Fs.sum(0, keepdim=True)
        Fs = Fs_.sum(0, keepdim=True)

        r = F.normalize(Fs - c_Fs, p=2)

        # working on support image to find the contribution of seen classes to new class
        fs = Fs * r
        Fs_ori = [x * r for x in Fs_ori]

        list_fs_ = []
        list_iou = []

        f = nn.Parameter(fs)
        #optimizer = torch.optim.SGD([f], lr=1e2)
        optimizer = torch.optim.Adam([f], lr=1e-2)
        
        # use gradient boosting to find fs_
        for i in range(step+1):
            with torch.enable_grad():
                list_fs_.append(f.clone())

                loss = 0
                list_x = []

                for Fs, Fl, xs_, ms_ in zip(Fs_ori, list_Fl_ori, xs, ms):
                    s = F.relu(F.cosine_similarity(Fs, f)).unsqueeze(1)
                    s = F.interpolate(s, Fl.shape[-2:], mode='bilinear', align_corners=True)
                    s_max = F.adaptive_max_pool2d(s, 1)
                    s = s / (s_max + 1e-16)

                    x = torch.cat([Fl * s,], 1)

                    x = self.exit_layer(x)

                    x = F.interpolate(x, xs_.shape[-2:], mode='bilinear', align_corners=True)

                    loss += F.cross_entropy(x, ms_)

                    list_x.append(x)

                # self.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                list_x = [x.argmax(1) for x in list_x]
            
                iou = torch.stack([((x == 1) & (ms_ == 1)).sum().float() / ((x == 1) | (ms_ == 1)).sum().float() 
                                for x, ms_ in zip(list_x, ms)]).mean()

                list_iou.append(iou)   
        
        list_iou = torch.stack(list_iou)

        list_fs_ = torch.cat(list_fs_)

        ###############################################################
        # working on query image and use list_iou as contribution to find the final mask
        s = F.relu(F.cosine_similarity((Fq * r).unsqueeze(1), (list_fs_).unsqueeze(0), dim=2))

        s = F.interpolate(s, Fl.shape[-2:], mode='bilinear', align_corners=True)
        s_max = F.adaptive_max_pool2d(s, 1)
        s = s / (s_max + 1e-16)
        s = s.permute(1, 0, 2, 3)

        x = torch.cat([Fl * s], 1)

        x = self.exit_layer(x)

        x = F.interpolate(x, xq.shape[-2:], mode='bilinear', align_corners=True)

        x = x.argmax(1)
        
        list_x = x.float()

        list_iou1 = ((x == 1) & (mq == 1)).sum(-1).sum(-1).float() / ((x == 1) | (mq == 1)).sum(-1).sum(-1).float()

        final_x1 = (list_x * list_iou1[..., None, None]).sum(0, keepdim=True)
        final_x = (list_x * list_iou[..., None, None]).sum(0, keepdim=True)

        return r, final_x, final_x1, None


__all__ = [
    'VGG', 'vgg16', 'resnet101'
]


model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]


class VGG(FewShotModel):

    def __init__(self, features, model=0, num_shots=1):
        super(VGG, self).__init__(model, num_shots)

        self.features = features

        if model in [0, 2, 3, 5, 8, 9]:
            self.exit_layer = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, dilation=1,  padding=1),  
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, kernel_size=1)
            )
        else:
            self.exit_layer = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, dilation=1,  padding=1),  
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, kernel_size=1)
            )

    def forward_all(self, x):

        for i in range(len(self.features)):
            if i != len(self.features) - 2:
                x = self.features[i](x)
                if i == 13:
                    x1 = x

        return x1, None, None, x


def make_layers(cfg, dilation=None, batch_norm=False, in_channels=3):
    layers = []
    for i, (v, d) in enumerate(zip(cfg, dilation)):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'D_deeplab': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
}

dilation = {
    'D': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 2, 2, 2, 'N', 4, 4, 4, 'N']
}


def vgg16(pretrained=False, in_channels=3, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D_deeplab'], dilation=dilation['D'], in_channels=in_channels, batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']), strict=False)
    return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm=None, last=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last:
            out = self.relu(out)

        return out


class ResNet(FewShotModel):

    def __init__(self, block, layers, num_groups=None, beta=False, os=8, model=0, num_shots=1):
        super().__init__(model, num_shots)
        self.inplanes = 64
        self._norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)

        if os == 16:
            strides = [2, 1]
            dilations = [1, 2]
        else:
            strides = [1, 1]
            dilations = [2, 4]

        if not beta:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self._norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[0], dilation=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[1], dilation=dilations[1], last=True)

        if model in [0, 2, 3, 5, 8, 9]:
            self.exit_layer = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, dilation=1,  padding=1),  
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, kernel_size=1)
            )
        else:
            self.exit_layer = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, dilation=1,  padding=1),  
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, kernel_size=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, last=False):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation//2), bias=False),
                self._norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation//2), norm=self._norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm=self._norm, last=i==blocks-1 and last))

        return nn.Sequential(*layers)

    def forward_all(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model
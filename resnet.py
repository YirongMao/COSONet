import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch
import layer_utils

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, ave_size=7, num_output=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(ave_size, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.num_output = num_output
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x) # B*256*14*14
        # conv = self.layer4(x)
        #
        # x = self.avgpool(conv)
        # feat = x.view(x.size(0), -1)
        # x = self.fc(feat)
        # if self.num_output == 3:
        #     return conv, feat, x
        # else:
        #     return x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        conv_4 = self.layer3(x)  # B*256*14*14
        conv = self.layer4(conv_4)

        x = self.avgpool(conv)
        feat = x.view(x.size(0), -1)
        x = self.fc(feat)
        if self.num_output == 3:
            return conv_4, conv, feat, x
        else:
            return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


class sqrt_el(nn.Module):
    def __init__(self):
        super(sqrt_el, self).__init__()
        self.feat_dim = 256

    def forward(self, cov):
        cov = torch.mul(torch.sign(cov), torch.sqrt(torch.abs(cov)))
        return cov


class sqrt_ns(nn.Module):
    def __init__(self, norm_type='T', num_iter=5):
        super(sqrt_ns, self).__init__()
        self.num_iter = num_iter
        self.norm_type = norm_type

        print('norm_type %s num_iter %s' %(norm_type, str(num_iter)))
        if norm_type == 'AT' or norm_type == 'AF':
            self.sqrtm = layer_utils.Sqrtm_autograd(norm_type=norm_type, num_iter=num_iter)

    def forward(self, A):
        if self.norm_type == 'T':
            sA = layer_utils.SqrtmLayer(A, self.num_iter)
            return sA
        else:
            sA = self.sqrtm(A)
            return sA


class Second_order(nn.Module):
    def __init__(self, agg_type='G'):
        super(Second_order, self).__init__()
        self.agg_type = agg_type

    def forward(self, x):
        '''
        :param x: N*C*H*W
        :param dim:
        :param use_cuda:
        :return: N*C*C
        '''
        cov = []
        if self.agg_type == 'C':
            '''
            refer to Eq. (1)
            '''
            x = x.view(x.size(0), x.size(1), -1)
            num_sample = x.size(-1)
            mean = torch.mean(x, dim=-1, keepdim=True)  # N*C*1
            cx = x - mean
            cxt = torch.transpose(cx, dim0=1, dim1=2)
            cov = (1.0 / num_sample) * torch.bmm(cx, cxt)  # N*C*C

        if self.agg_type == 'B':
            x = x.view(x.size(0), x.size(1), -1)
            num_sample = x.size(-1)
            xt = torch.transpose(x, dim0=1, dim1=2)
            cov = (1.0/num_sample)*torch.bmm(x, xt) # N*C*C

        if self.agg_type == 'G':
            # print(self.agg_type)
            x = x.view(x.size(0), x.size(1), -1)
            num_sample = x.size(-1)
            xt = torch.transpose(x, dim0=1, dim1=2)
            cov = (1.0 / num_sample) * torch.bmm(x, xt)  # N*C*C
            mean = torch.mean(x, dim=-1, keepdim=True)  # N*C*1
            mean_t = torch.transpose(mean, dim0=1, dim1=2)  # N*1*C

            cov = torch.cat((cov, mean), dim=-1)  # N*C*(C+1)
            one = Variable(torch.ones((cov.size(0), 1, 1))).cuda()
            mean_t = torch.cat((mean_t, one), dim=-1)  # N*1*(C+1)
            cov = torch.cat((cov, mean_t), dim=1)

        return cov


class SPD_Transfer(nn.Module):
    def __init__(self, input_feat_dim, feat_dim=64, non_linear='relu', num_iter=5, norm_type='T'):
        super(SPD_Transfer, self).__init__()
        param = torch.FloatTensor(feat_dim, input_feat_dim)
        torch.nn.init.xavier_normal(param)
        self.w = torch.nn.Parameter(param)
        self.use_non_linear = False

        if len(non_linear) > 0:
            self.use_non_linear = True
            if non_linear == 'sqrt_mat':
                self.non_linear_func = sqrt_ns(num_iter=num_iter, norm_type=norm_type)
            elif non_linear == 'sqrt_el':
                self.non_linear_func = sqrt_el()
            else:
                self.non_linear_func = nn.ReLU()

        print('nonlinear %s %s' % (str(non_linear), str(self.use_non_linear)))

    def forward(self, cov):
        # 2D-FC layer, please refer to Eq. (5)
        W = torch.unsqueeze(self.w, dim=0)
        WX = torch.matmul(W, cov)
        cov = torch.matmul(WX, torch.transpose(W, 1, 2))
        if self.use_non_linear:
            cov = self.non_linear_func(cov)
        return cov


class Gaussian_embedding(nn.Module):
    def __init__(self, agg_type='C', pre_conv=True, input_conv_feat_dim=512,
                 conv_feat_dim=256, num_iter=5, norm_type='T'):
        '''

        :param agg_type G: Gaussian Embedding, B: bilinear pooling, C: covariance pooling
        :param pre_conv:
        :param input_conv_feat_dim:
        :param conv_feat_dim:
        '''
        super(Gaussian_embedding, self).__init__()
        self.second_order = Second_order(agg_type)
        sqrt_dim = conv_feat_dim
        if agg_type == 'G':
            sqrt_dim += 1
        self.sqrt_func = sqrt_ns(num_iter=num_iter, norm_type=norm_type)
        self.pre_conv = pre_conv
        if pre_conv:
            self.conv2d = nn.Conv2d(input_conv_feat_dim, conv_feat_dim, kernel_size=1)
            self.bn = nn.BatchNorm2d(conv_feat_dim)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        if self.pre_conv:
            input = self.conv2d(input)
            input = self.bn(input)
            input = self.relu(input)

        cov_p = self.second_order(input)
        cov = self.sqrt_func(cov_p)
        return input, cov_p, cov


class ResNet_Cov_pre(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_Cov_pre, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool_14 = nn.AvgPool2d(14, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        conv_4 = self.layer3(x) # B*256*14*14

        return conv_4


class ResNet_Cov_pro(nn.Module):
    def __init__(self, block, layers, agg_type, input_conv_feat_dim=512,
                 num_iter=5, norm_type='T', num_classes=1000):
        super(ResNet_Cov_pro, self).__init__()
        self.inplanes = 256
        self.layer_e = self._make_layer(block, input_conv_feat_dim, layers)
        self.GE = Gaussian_embedding(agg_type=agg_type, pre_conv=True,
                                     input_conv_feat_dim=input_conv_feat_dim, conv_feat_dim=256,
                                     num_iter=num_iter, norm_type=norm_type)
        non_linear = ''
        # 2D FC layer
        self.spd_transfer = SPD_Transfer(256, 64, non_linear=non_linear, num_iter=num_iter, norm_type=norm_type) #relu or sqrt
        self.fc = nn.Linear(64 * 64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, conv):
        # the last convolution block
        conv = self.layer_e(conv)
        # compression convolution layer, second-order pooling, matrix square root
        conv_f, cov_p, cov = self.GE(conv)
        # 2D-FC
        cov = self.spd_transfer(cov)
        # Flatten
        cov_feat = cov.view((cov.size(0), -1))
        # logits
        x = self.fc(cov_feat)
        return conv_f, cov_feat, x


def resnet34_cov(agg_type='C', num_classes=1000, num_iter=5, norm_type='T'):

    # the first 4 convolution blocks
    layers = [3, 4, 6, 3]
    conv_model = ResNet_Cov_pre(BasicBlock, layers, num_classes=num_classes)

    # the last convolution block, compression convolution layer,
    # second-order pooling, matrix square root, 2D-FC
    spd_model = ResNet_Cov_pro(agg_type=agg_type, num_iter=num_iter, norm_type=norm_type, block=BasicBlock,
                               layers=3, input_conv_feat_dim=512, num_classes=num_classes)
    return conv_model, spd_model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    inputs = Variable(torch.randn((1, 3, 224, 224)))
    model = resnet18()
    out = model(inputs)
    print('w')

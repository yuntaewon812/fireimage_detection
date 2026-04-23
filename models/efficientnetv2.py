import copy
from functools import partial
from collections import OrderedDict
import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    """Convolution-Normalization-Activation Module"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )


class SEUnit(nn.Module):
    """Squeeze-Excitation Unit

    paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

    """
    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    """StochasticDepth

    paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39

    :arg
        - prob: Probability of dying
        - mode: "row" or "all". "row" means that each row survives with different probability
    """
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class MBConvConfig:
    """EfficientNet Building block configuration"""
    def __init__(self, expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int,
                 use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    """EfficientNet main building blocks

    :arg
        - c: MBConvConfig instance
        - sd_prob: stochastic path probability
    """
    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
        elif c.fused:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
            block.append(('fused_point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
        else:
            block.append(('linear_bottleneck', ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act)))
            block.append(('depth_wise', ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer, c.act)))
            block.append(('se', SEUnit(inter_channel, 4 * c.expand_ratio)))
            block.append(('point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class EfficientNetV2(nn.Module):
    """Pytorch Implementation of EfficientNetV2

    paper: https://arxiv.org/abs/2104.00298

    - reference 1 (pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    - reference 2 (official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py

    :arg
        - layer_infos: list of MBConvConfig
        - out_channels: bottleneck channel
        - nlcass: number of class
        - dropout: dropout probability before classifier layer
        - stochastic depth: stochastic depth probability
    """
    def __init__(self, layer_infos, out_channels=1280, nclass=0, dropout=0.2, stochastic_depth=0.0,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
        self.head = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct(self.final_stage_channel, out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass) if nclass else nn.Identity())
        ]))

    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head.bottleneck(x)
        x = self.head.avgpool(x)
        x = self.head.flatten(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head.dropout(x)
        x = self.head.classifier(x)
        return x

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)


def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


def get_efficientnet_v2_structure(model_name):
    """Get EfficientNetV2 model structure configurations"""
    if 'efficientnet_v2_s' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
    elif 'efficientnet_v2_m' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]
    elif 'efficientnet_v2_l' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 7, False, True),
            (4, 3, 2, 64, 96, 7, False, True),
            (4, 3, 2, 96, 192, 10, True, False),
            (6, 3, 1, 192, 224, 19, True, False),
            (6, 3, 2, 224, 384, 25, True, False),
            (6, 3, 1, 384, 640, 7, True, False),
        ]
    elif 'efficientnet_v2_xl' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 8, False, True),
            (4, 3, 2, 64, 96, 8, False, True),
            (4, 3, 2, 96, 192, 16, True, False),
            (6, 3, 1, 192, 256, 24, True, False),
            (6, 3, 2, 256, 512, 32, True, False),
            (6, 3, 1, 512, 640, 8, True, False),
        ]


def get_efficientnet_v2(model_name, pretrained=False, nclass=0, dropout=0.1, stochastic_depth=0.2, **kwargs):
    """Create EfficientNetV2 model"""
    residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(model_name)]
    model = EfficientNetV2(residual_config, 1280, nclass, dropout=dropout, stochastic_depth=stochastic_depth, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)
    return model


class EfficientNetV2ForImageClassification(nn.Module):
    """EfficientNetV2 wrapper for image classification compatible with the project"""
    
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512, 
                 model_variant='s'):
        super(EfficientNetV2ForImageClassification, self).__init__()
        
        # Model configurations for different variants
        configs = {
            's': {
                'model_name': 'efficientnet_v2_s',
                'dropout': 0.2,
                'stochastic_depth': 0.2
            },
            'm': {
                'model_name': 'efficientnet_v2_m',
                'dropout': 0.3,
                'stochastic_depth': 0.3
            },
            'l': {
                'model_name': 'efficientnet_v2_l',
                'dropout': 0.4,
                'stochastic_depth': 0.4
            },
            'xl': {
                'model_name': 'efficientnet_v2_xl',
                'dropout': 0.4,
                'stochastic_depth': 0.5
            }
        }
        
        config = configs.get(model_variant, configs['s'])
        
        # Create backbone without classification head
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=False,
            nclass=0,  # No classification head in backbone
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth']
        )
        
        # Replace classifier with identity
        self.backbone.head.classifier = nn.Identity()
        
        # Calculate output dimension - EfficientNetV2 uses 1280 as bottleneck
        num_features = 1280
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        # Extract features
        features = self.backbone.forward_features(images)
        
        # Classification
        logits = self.classifier(features)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

from models.module import SAFEModule

class EfficientNetV2ForImageClassification_v2(nn.Module):
    """EfficientNetV2 wrapper for image classification compatible with the project"""

    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='s'):
        super(EfficientNetV2ForImageClassification_v2, self).__init__()

        # Model configurations for different variants
        configs = {
            's': {
                'model_name': 'efficientnet_v2_s',
                'dropout': 0.2,
                'stochastic_depth': 0.2
            },
            'm': {
                'model_name': 'efficientnet_v2_m',
                'dropout': 0.3,
                'stochastic_depth': 0.3
            },
            'l': {
                'model_name': 'efficientnet_v2_l',
                'dropout': 0.4,
                'stochastic_depth': 0.4
            },
            'xl': {
                'model_name': 'efficientnet_v2_xl',
                'dropout': 0.4,
                'stochastic_depth': 0.5
            }
        }

        config = configs.get(model_variant, configs['s'])

        # Create backbone without classification head
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=False,
            nclass=0,  # No classification head in backbone
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth']
        )

        # Replace classifier with identity
        self.backbone.head.classifier = nn.Identity()

        # Calculate output dimension - EfficientNetV2 uses 1280 as bottleneck
        num_features = 1280

        self.safe_module = SAFEModule(num_features)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        # Extract features
        features = self.backbone.forward_features(images)

        # Classification
        logits = self.classifier(features)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

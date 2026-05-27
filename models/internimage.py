import math
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().div_(keep)
        return x * noise


class DCNv3(nn.Module):
    """Deformable Convolutional Network v3.

    paper: https://arxiv.org/abs/2211.05778
    Uses group-wise deformable convolution with modulation masks.
    Each group learns independent spatial offsets and masks.
    """

    def __init__(self, channels, kernel_size=3, group=4, offset_scale=1.0):
        super().__init__()
        K = kernel_size
        assert channels % group == 0, "channels must be divisible by group"
        self.channels = channels
        self.K = K
        self.G = group
        self.Gc = channels // group

        # Depthwise conv for offset/mask generation
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        # Offset (2D per kernel position per group) and modulation mask
        self.offset_proj = nn.Linear(channels, group * K * K * 2)
        self.mask_proj = nn.Linear(channels, group * K * K)

        # Per-group conv weights: weight[g] maps Gc → Gc channels
        self.weight = nn.Parameter(torch.empty(group, self.Gc, self.Gc, K, K))
        self.bias = nn.Parameter(torch.zeros(channels))

        self.value_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self.offset_scale = offset_scale

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.zeros_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)
        for g in range(self.G):
            nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        x_c = x.permute(0, 3, 1, 2).contiguous()            # (B, C, H, W)
        dw_out = self.dw_conv(x_c).permute(0, 2, 3, 1)      # (B, H, W, C)

        offset = self.offset_proj(dw_out) * self.offset_scale  # (B, H, W, G*K*K*2)
        mask = self.mask_proj(dw_out).sigmoid()                 # (B, H, W, G*K*K)

        v = self.value_proj(x).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        offset = offset.permute(0, 3, 1, 2).contiguous()          # (B, G*K*K*2, H, W)
        mask = mask.permute(0, 3, 1, 2).contiguous()              # (B, G*K*K, H, W)

        K = self.K
        outputs = []
        for g in range(self.G):
            v_g = v[:, g * self.Gc:(g + 1) * self.Gc]              # (B, Gc, H, W)
            w_g = self.weight[g]                                    # (Gc, Gc, K, K)
            b_g = self.bias[g * self.Gc:(g + 1) * self.Gc]         # (Gc,)
            off_g = offset[:, g * K * K * 2:(g + 1) * K * K * 2]  # (B, K*K*2, H, W)
            mask_g = mask[:, g * K * K:(g + 1) * K * K]            # (B, K*K, H, W)

            out_g = deform_conv2d(v_g, off_g, w_g, b_g,
                                  stride=1, padding=K // 2, mask=mask_g)
            outputs.append(out_g)

        out = torch.cat(outputs, dim=1)        # (B, C, H, W)
        out = out.permute(0, 2, 3, 1)          # (B, H, W, C)
        return self.output_proj(out)


class InternImageBlock(nn.Module):
    """InternImage basic block: DCNv3 + FFN with residual connections."""

    def __init__(self, channels, mlp_ratio=4.0, drop=0.0, drop_path=0.0, group=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.dcnv3 = DCNv3(channels, group=group)

        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, channels),
            nn.Dropout(drop),
        )
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        # x: (B, H, W, C)
        x = x + self.drop_path(self.dcnv3(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Strided conv downsampling between stages (2x)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        # x: (B, H, W, C)
        x = self.norm(x).permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.conv(x).permute(0, 2, 3, 1)  # (B, H/2, W/2, out_ch)


class InternImage(nn.Module):
    """InternImage: Exploring Large-Scale Vision Foundation Models
    with Deformable Convolutions (CVPR 2023).

    paper: https://arxiv.org/abs/2211.05778
    Stem downsamples 224→56 (stride 4), then each stage halves spatial dims.
    Gc (channels per group) is kept constant at 16 across all stages.
    """

    def __init__(self, base_channels=64, depths=(4, 4, 18, 4),
                 groups=(4, 8, 16, 32), mlp_ratio=4.0,
                 drop=0.0, drop_path_rate=0.2):
        super().__init__()

        ch = [base_channels * (2 ** i) for i in range(len(depths))]

        # Stride-4 stem: 224 → 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1, bias=False),
            nn.GELU(),
        )
        self.stem_norm = nn.LayerNorm(ch[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        idx = 0
        for stage_id, (d, g) in enumerate(zip(depths, groups)):
            stage = nn.ModuleList([
                InternImageBlock(ch[stage_id], mlp_ratio, drop, dpr[idx + i], g)
                for i in range(d)
            ])
            self.stages.append(stage)
            idx += d
            if stage_id < len(depths) - 1:
                self.downsample.append(PatchMerging(ch[stage_id], ch[stage_id + 1]))

        self.num_features = ch[-1]
        self.norm = nn.LayerNorm(self.num_features)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x).permute(0, 2, 3, 1)  # (B, 56, 56, C)
        x = self.stem_norm(x)

        for stage_id, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            if stage_id < len(self.stages) - 1:
                x = self.downsample[stage_id](x)

        x = self.norm(x)
        return x.mean(dim=[1, 2])  # global average pooling → (B, C)

    def forward(self, x):
        return self.forward_features(x)


from models.module import SAFEModule


class InternImageForImageClassification(nn.Module):
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='tiny'):
        super().__init__()
        configs = {
            'tiny':  dict(base_channels=64,  depths=(4, 4, 18, 4),
                          groups=(4, 8, 16, 32), drop_path_rate=0.2),
            'small': dict(base_channels=80,  depths=(4, 4, 21, 4),
                          groups=(5, 10, 20, 40), drop_path_rate=0.3),
            'base':  dict(base_channels=112, depths=(4, 4, 21, 4),
                          groups=(7, 14, 28, 56), drop_path_rate=0.4),
        }
        cfg = configs.get(model_variant, configs['tiny'])
        self.backbone = InternImage(**cfg)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images, labels=None):
        features = self.backbone.forward_features(images)
        logits = self.classifier(features)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits


class InternImageForImageClassification_v2(nn.Module):
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='tiny'):
        super().__init__()
        configs = {
            'tiny':  dict(base_channels=64,  depths=(4, 4, 18, 4),
                          groups=(4, 8, 16, 32), drop_path_rate=0.2),
            'small': dict(base_channels=80,  depths=(4, 4, 21, 4),
                          groups=(5, 10, 20, 40), drop_path_rate=0.3),
            'base':  dict(base_channels=112, depths=(4, 4, 21, 4),
                          groups=(7, 14, 28, 56), drop_path_rate=0.4),
        }
        cfg = configs.get(model_variant, configs['tiny'])
        self.backbone = InternImage(**cfg)
        num_features = self.backbone.num_features
        self.safe_module = SAFEModule(num_features)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images, labels=None):
        features = self.backbone.forward_features(images)
        logits = self.classifier(features)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

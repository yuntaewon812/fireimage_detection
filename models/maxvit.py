import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x).flatten(1)
        s = self.sigmoid(self.fc2(self.act(self.fc1(s))))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, se_ratio=0.25, stride=1, drop_path=0.0):
        super().__init__()
        mid_ch = in_ch * expansion
        self.use_skip = stride == 1 and in_ch == out_ch
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, mid_ch, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch),
            nn.GELU(),
            SqueezeExcitation(mid_ch, se_ratio),
            nn.Conv2d(mid_ch, out_ch, 1),
        )
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.conv(x)
        return x + self.drop_path(out) if self.use_skip else out


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, window_size):
        super().__init__()
        self.ws = window_size
        self.table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))
        flat = coords.flatten(1)
        rel = flat[:, :, None] - flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_idx', rel.sum(-1))

    def forward(self):
        N = self.ws ** 2
        bias = self.table[self.rel_idx.view(-1)].view(N, N, -1)
        return bias.permute(2, 0, 1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_bias = RelativePositionBias(num_heads, window_size) if window_size else None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_bias is not None:
            attn = attn + self.rel_bias()
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MaxViTBlock(nn.Module):
    """MBConv + Block (local window) Attention + Grid (dilated) Attention."""

    def __init__(self, in_ch, out_ch, num_heads, partition_size=7,
                 mlp_ratio=4.0, drop_path=0.0, attn_drop=0.0, drop=0.0, stride=1):
        super().__init__()
        self.p = partition_size
        self.mbconv = MBConv(in_ch, out_ch, stride=stride, drop_path=drop_path)
        dim = out_ch
        mlp_hid = int(dim * mlp_ratio)

        # Block (local) attention
        self.norm1 = nn.LayerNorm(dim)
        self.block_attn = Attention(dim, num_heads, window_size=partition_size,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.block_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hid, dim), nn.Dropout(drop),
        )
        self.dp1 = StochasticDepth(drop_path)
        self.dp2 = StochasticDepth(drop_path)

        # Grid (dilated) attention
        self.norm3 = nn.LayerNorm(dim)
        self.grid_attn = Attention(dim, num_heads, window_size=None,
                                   attn_drop=attn_drop, proj_drop=drop)
        self.norm4 = nn.LayerNorm(dim)
        self.grid_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hid, dim), nn.Dropout(drop),
        )
        self.dp3 = StochasticDepth(drop_path)
        self.dp4 = StochasticDepth(drop_path)

    # ---- partition helpers ----

    def _pad(self, x):
        """H, W를 window size p의 배수로 패딩."""
        B, H, W, C = x.shape
        p = self.p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            # F.pad: (last_dim_left, last_dim_right, ...) 순서 → C 무시, W, H 패딩
            x = x.permute(0, 3, 1, 2)           # (B, C, H, W)
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (B, C, H+pad_h, W+pad_w)
            x = x.permute(0, 2, 3, 1)           # (B, H+pad_h, W+pad_w, C)
        return x, pad_h, pad_w

    def _block_part(self, x):
        x, pad_h, pad_w = self._pad(x)
        B, H, W, C = x.shape
        p, gh, gw = self.p, H // self.p, W // self.p
        x = x.reshape(B, gh, p, gw, p, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(B * gh * gw, p * p, C), (B, gh, gw, pad_h, pad_w)

    def _block_unpart(self, x, meta, H, W):
        B, gh, gw, pad_h, pad_w = meta
        p = self.p
        x = x.reshape(B, gh, gw, p, p, -1).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, gh * p, gw * p, -1)
        return x[:, :H, :W, :].contiguous()

    def _grid_part(self, x):
        x, pad_h, pad_w = self._pad(x)
        B, H, W, C = x.shape
        p, gh, gw = self.p, H // self.p, W // self.p
        x = x.reshape(B, gh, p, gw, p, C).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(B * p * p, gh * gw, C), (B, p, gh, gw, pad_h, pad_w)

    def _grid_unpart(self, x, meta, H, W):
        B, p, gh, gw, pad_h, pad_w = meta
        x = x.reshape(B, p, p, gh, gw, -1).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, gh * p, gw * p, -1)
        return x[:, :H, :W, :].contiguous()

    def forward(self, x):
        x = self.mbconv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Block attention
        xp, meta = self._block_part(x)
        xp = xp + self.dp1(self.block_attn(self.norm1(xp)))
        xp = xp + self.dp2(self.block_mlp(self.norm2(xp)))
        x = self._block_unpart(xp, meta, H, W)

        # Grid attention
        xg, meta = self._grid_part(x)
        xg = xg + self.dp3(self.grid_attn(self.norm3(xg)))
        xg = xg + self.dp4(self.grid_mlp(self.norm4(xg)))
        x = self._grid_unpart(xg, meta, H, W)

        return x.permute(0, 3, 1, 2)  # (B, C, H, W)


class MaxViT(nn.Module):
    """MaxViT: Multi-Axis Vision Transformer (ECCV 2022).

    paper: https://arxiv.org/abs/2204.01697
    partition_size=7 divides 224→112→56→28→14 at each stage.
    """

    def __init__(self, img_size=224, stem_ch=64,
                 channels=(64, 128, 256, 512), depths=(2, 2, 5, 2),
                 num_heads=(2, 4, 8, 16), partition_size=7,
                 mlp_ratio=4.0, drop_path_rate=0.2, attn_drop=0.0, drop=0.0,
                 num_classes=0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1, bias=False),
        )

        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        # Stage 0 keeps spatial size; stages 1-3 downsample with stride-2 first block
        strides = [1, 2, 2, 2]

        self.stages = nn.ModuleList()
        prev, idx = stem_ch, 0
        for ch, d, heads, stride in zip(channels, depths, num_heads, strides):
            blocks = []
            for i in range(d):
                blocks.append(MaxViTBlock(
                    in_ch=prev if i == 0 else ch,
                    out_ch=ch,
                    num_heads=heads,
                    partition_size=partition_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[idx],
                    attn_drop=attn_drop,
                    drop=drop,
                    stride=stride if i == 0 else 1,
                ))
                idx += 1
            self.stages.append(nn.Sequential(*blocks))
            prev = ch

        self.num_features = channels[-1]
        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
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
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return x.mean(dim=[1, 2])  # global average pooling

    def forward(self, x):
        return self.head(self.forward_features(x))


from models.module import SAFEModule


class MaxViTForImageClassification(nn.Module):
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='tiny'):
        super().__init__()
        configs = {
            'tiny':  dict(stem_ch=64, channels=(64, 128, 256, 512),
                          depths=(2, 2, 5, 2), num_heads=(2, 4, 8, 16), drop_path_rate=0.2),
            'small': dict(stem_ch=64, channels=(96, 192, 384, 768),
                          depths=(2, 2, 5, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.3),
            'base':  dict(stem_ch=64, channels=(96, 192, 384, 768),
                          depths=(2, 6, 14, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.4),
        }
        cfg = configs.get(model_variant, configs['tiny'])
        self.backbone = MaxViT(img_size=img_size, partition_size=7, num_classes=0, **cfg)
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


class MaxViTForImageClassification_v2(nn.Module):
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='tiny'):
        super().__init__()
        configs = {
            'tiny':  dict(stem_ch=64, channels=(64, 128, 256, 512),
                          depths=(2, 2, 5, 2), num_heads=(2, 4, 8, 16), drop_path_rate=0.2),
            'small': dict(stem_ch=64, channels=(96, 192, 384, 768),
                          depths=(2, 2, 5, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.3),
            'base':  dict(stem_ch=64, channels=(96, 192, 384, 768),
                          depths=(2, 6, 14, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.4),
        }
        cfg = configs.get(model_variant, configs['tiny'])
        self.backbone = MaxViT(img_size=img_size, partition_size=7, num_classes=0, **cfg)
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

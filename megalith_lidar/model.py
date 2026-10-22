from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from megalith_lidar.terrain_features import NUM_TERRAIN_CHANNELS


class SCSEBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation attention block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.c_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.s_se = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.c_se(x) + x * self.s_se(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            SCSEBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MegalithUNet(nn.Module):
    """
    Multi-task ResNet34-UNet with reflection padding (zero boundary-induced center bias)
    and Multiple-Instance Learning (MIL) spatial peak pooling.
    """

    def __init__(
        self,
        in_channels: int = NUM_TERRAIN_CHANNELS,
        num_classes: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            padding_mode="reflect",
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                rep = (in_channels + 2) // 3
                w = old_conv.weight.repeat(1, rep, 1, 1)[:, :in_channels, :, :] * (3.0 / in_channels)
                new_conv.weight.copy_(w)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self.stem = nn.Sequential(
            new_conv,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.full_skip = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 96)
        self.dec1 = DecoderBlock(96, 64, 64)
        self.dec0 = DecoderBlock(64, 32, 48)

        self.seg_head = nn.Conv2d(48, 1, kernel_size=1)
        self.cls_seg_head = nn.Conv2d(48, num_classes, kernel_size=1)

        self.e4_proj = nn.Sequential(
            nn.Linear(512, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.d0_proj = nn.Sequential(
            nn.Linear(96, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.tile_mlp = nn.Sequential(
            nn.Linear(72, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.30),
        )
        self.tile_presence_out = nn.Linear(64, 1)
        self.tile_type_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s0 = self.full_skip(x)
        s1 = self.stem(x)
        e1 = self.enc1(self.maxpool(s1))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        seg_logits = self.seg_head(d0)
        cls_seg_logits = self.cls_seg_head(d0)

        b = x.shape[0]
        e4_vec = self.e4_proj(F.adaptive_avg_pool2d(e4, 1).flatten(1))
        d0_cat = torch.cat(
            [F.adaptive_avg_pool2d(d0, 1).flatten(1), F.adaptive_max_pool2d(d0, 1).flatten(1)],
            dim=1,
        )
        d0_vec = self.d0_proj(d0_cat)

        flat_seg = seg_logits.view(b, -1)
        topk_256, _ = torch.topk(flat_seg, k=min(256, flat_seg.shape[1]), dim=1)
        seg_q1 = topk_256[:, :16].mean(dim=1, keepdim=True)
        seg_q2 = topk_256[:, :64].mean(dim=1, keepdim=True)
        seg_q3 = topk_256.mean(dim=1, keepdim=True)
        seg_mean = flat_seg.mean(dim=1, keepdim=True)

        flat_cls = cls_seg_logits.view(b, cls_seg_logits.shape[1], -1)
        cls_peaks = torch.topk(flat_cls, k=min(64, flat_cls.shape[2]), dim=2)[0].mean(dim=2)

        feat_vec = torch.cat([e4_vec, d0_vec, seg_q1, seg_q2, seg_q3, seg_mean, cls_peaks], dim=1)
        hidden = self.tile_mlp(feat_vec)
        tile_logits = 0.55 * self.tile_presence_out(hidden) + 0.45 * seg_q2
        type_logits = self.tile_type_out(hidden) + 0.35 * cls_peaks

        return {
            "seg_logits": seg_logits,
            "cls_seg_logits": cls_seg_logits,
            "tile_logits": tile_logits,
            "type_logits": type_logits,
            "seg_peak_logit": seg_q2,
        }


class MegalithLoss(nn.Module):
    """
    Statistically balanced Segmentation + Multiple-Instance Peak Suppression + Tile Classification loss.
    Explicitly penalizes false-positive spatial peaks on negative tiles (MIL objective) so background
    landscapes do not trigger false alarms during large-area search.
    """

    def __init__(self, pos_pixel_weight: float = 2.5, pos_tile_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_pixel_weight = pos_pixel_weight
        self.pos_tile_weight = pos_tile_weight
        self.register_buffer("cls_weights", torch.tensor([0.5, 1.3, 1.3, 1.8], dtype=torch.float32))
        self.register_buffer("type_weights", torch.tensor([1.0, 1.2, 1.2, 1.2], dtype=torch.float32))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        target_tile: torch.Tensor,
        target_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        seg_logits = outputs["seg_logits"].squeeze(1)  # (B, H, W)
        cls_seg_logits = outputs["cls_seg_logits"]  # (B, C, H, W)
        tile_logits = outputs["tile_logits"].squeeze(1)  # (B,)
        type_logits = outputs["type_logits"]  # (B, C)

        valid_px = target_mask != 255
        valid_f = valid_px.float()
        bin_mask = ((target_mask > 0) & valid_px).float()

        # 1. Pixel Focal + BCE with moderate pos_pixel_weight, ignoring legal-buffer pixels (255)
        pw = torch.tensor(self.pos_pixel_weight, device=seg_logits.device, dtype=seg_logits.dtype)
        bce_pixel = F.binary_cross_entropy_with_logits(seg_logits, bin_mask, pos_weight=pw, reduction="none")
        probs = torch.sigmoid(seg_logits)
        pt = torch.where(bin_mask > 0.5, probs, 1.0 - probs)
        focal_map = (1.0 - pt).pow(2.0) * bce_pixel * valid_f
        focal_loss = focal_map.sum() / valid_f.sum().clamp(min=1.0)

        # 2. Soft Dice loss on positive tiles only (excluding IGNORE_INDEX=255 legal-buffer pixels)
        probs_valid = probs * valid_f
        inter = (probs_valid * bin_mask).sum(dim=(1, 2))
        union = probs_valid.sum(dim=(1, 2)) + bin_mask.sum(dim=(1, 2))
        pos_tile_mask = target_tile > 0.5
        if torch.any(pos_tile_mask):
            dice_loss = (1.0 - (2.0 * inter[pos_tile_mask] + 1.0) / (union[pos_tile_mask] + 1.0)).mean()
        else:
            dice_loss = probs.mean() * 0.0

        # 3. Multiple-Instance Learning (MIL) Peak Suppression Loss:
        # The top-64 spatial peak logit of a tile must match target_tile (0 on negative tiles, 1 on positive tiles)
        flat_seg = seg_logits.view(seg_logits.shape[0], -1)
        topk_64 = torch.topk(flat_seg, k=min(64, flat_seg.shape[1]), dim=1)[0].mean(dim=1)
        ptw = torch.tensor(self.pos_tile_weight, device=seg_logits.device, dtype=seg_logits.dtype)
        mil_peak_loss = F.binary_cross_entropy_with_logits(topk_64, target_tile.float(), pos_weight=ptw)

        # 4. Multi-class segmentation cross-entropy with ignore_index=255
        cls_w = self.cls_weights.to(device=cls_seg_logits.device, dtype=cls_seg_logits.dtype)
        ce_seg = F.cross_entropy(cls_seg_logits, target_mask.long(), weight=cls_w, ignore_index=255)

        # 5. Tile-level binary presence loss with prevalence correction
        tile_bce = F.binary_cross_entropy_with_logits(tile_logits, target_tile.float(), pos_weight=ptw)

        # 6. Tile-level multi-class type loss
        tw = self.type_weights.to(device=type_logits.device, dtype=type_logits.dtype)
        tile_ce = F.cross_entropy(type_logits, target_type.long(), weight=tw)

        total = 1.0 * focal_loss + 0.8 * dice_loss + 0.7 * mil_peak_loss + 0.5 * ce_seg + 0.8 * tile_bce + 0.4 * tile_ce
        metrics = {
            "loss": float(total.detach().item()),
            "focal": float(focal_loss.detach().item()),
            "dice": float(dice_loss.detach().item()),
            "mil_peak": float(mil_peak_loss.detach().item()),
            "tile_bce": float(tile_bce.detach().item()),
        }
        return total, metrics

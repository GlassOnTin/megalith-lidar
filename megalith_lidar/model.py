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

        # Differentiable D4-invariant 2D footprint geometry head (area_frac, log_elongation,
        # ring_hollowness, boundary_compactness). Zero-initialized output layer preserves exact
        # backwards compatibility when loading pre-v4 checkpoints with strict=False.
        self.geom_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.geom_tile_out = nn.Linear(16, 1)
        self.geom_type_out = nn.Linear(16, num_classes)
        nn.init.zeros_(self.geom_tile_out.weight)
        nn.init.zeros_(self.geom_tile_out.bias)
        nn.init.zeros_(self.geom_type_out.weight)
        nn.init.zeros_(self.geom_type_out.bias)

        # Dual-stream 16m Scale-Space Cropmark & Soilmark branch (channels 6..7: ΔExG_16m, ΔL_16m).
        # Accepts either 6-channel (LiDAR only) or 8-channel (LiDAR + Cropmarks) input tensors seamlessly.
        # Output projections use residual logits on seg_logits/cls_seg_logits/tile_logits so gradients
        # flow directly without disturbing frozen BatchNorm statistics inside dec0/dec1.
        self.cropmark_skip = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode="reflect", bias=True),
        )
        self.cropmark_stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="reflect", bias=True),
        )
        self.cropmark_seg_head = nn.Sequential(
            nn.Conv2d(54, 24, kernel_size=5, padding=2, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, dilation=2, padding=2, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 1 + num_classes, kernel_size=1, bias=True),
        )
        self.cropmark_mlp = nn.Sequential(
            nn.Linear(10, 24),
            nn.LayerNorm(24),
            nn.GELU(),
            nn.Linear(24, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.cropmark_tile_out = nn.Linear(16, 1)
        self.cropmark_type_out = nn.Linear(16, num_classes)
        nn.init.zeros_(self.cropmark_skip[-1].weight)
        nn.init.zeros_(self.cropmark_skip[-1].bias)
        nn.init.zeros_(self.cropmark_stem[-1].weight)
        nn.init.zeros_(self.cropmark_stem[-1].bias)
        nn.init.zeros_(self.cropmark_seg_head[-1].weight)
        nn.init.zeros_(self.cropmark_seg_head[-1].bias)
        nn.init.zeros_(self.cropmark_tile_out.weight)
        nn.init.zeros_(self.cropmark_tile_out.bias)
        nn.init.zeros_(self.cropmark_type_out.weight)
        nn.init.zeros_(self.cropmark_type_out.bias)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        own_state = self.state_dict()
        merged = dict(state_dict)
        for k, v in own_state.items():
            if (k.startswith("geom_") or k.startswith("cropmark_")) and k not in merged:
                merged[k] = v
        return super().load_state_dict(merged, strict=strict, assign=assign)

    @staticmethod
    def _compute_d4_geometric_moments(seg_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute 4 differentiable D4-equivariant/invariant 2D shape descriptors from `seg_logits` (B, 1, H, W):
          0. area_frac       : mean predicted foreground probability across the tile
          1. log_elongation  : log(sqrt(lambda_1 / lambda_2)) from probability-weighted 2D covariance
          2. ring_hollowness : outer-annulus probability minus central-core probability around centroid
          3. compactness     : 4 * pi * area / (perimeter^2 + eps) via total variation |grad(p)|
        """
        probs = torch.sigmoid(seg_logits.float()).squeeze(1)  # (B, H, W)
        b, h, w = probs.shape
        device = probs.device
        dtype = probs.dtype

        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w)

        area_frac = (probs.mean(dim=(1, 2))).unsqueeze(1)  # (B, 1)

        # Thresholded foreground weight so background pixels (p < 0.35) do not dilute 2D moments
        fg_w = F.relu(probs - 0.35).pow(2.0) + 1e-7
        w_sum = fg_w.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)

        cx = (fg_w * xs).sum(dim=(1, 2), keepdim=True) / w_sum
        cy = (fg_w * ys).sum(dim=(1, 2), keepdim=True) / w_sum
        dx = xs - cx
        dy = ys - cy

        cov_xx = (fg_w * dx * dx).sum(dim=(1, 2)) / w_sum.squeeze(-1).squeeze(-1) + 1e-5
        cov_yy = (fg_w * dy * dy).sum(dim=(1, 2)) / w_sum.squeeze(-1).squeeze(-1) + 1e-5
        cov_xy = (fg_w * dx * dy).sum(dim=(1, 2)) / w_sum.squeeze(-1).squeeze(-1)

        trace = cov_xx + cov_yy
        det = (cov_xx * cov_yy - cov_xy * cov_xy).clamp(min=1e-8)
        disc = torch.sqrt((trace * trace - 4.0 * det).clamp(min=0.0))
        lam1 = 0.5 * (trace + disc)
        lam2 = (0.5 * (trace - disc)).clamp(min=1e-6)
        log_elong = 0.5 * torch.log((lam1 / lam2).clamp(min=1.0, max=100.0)).unsqueeze(1)  # (B, 1)

        r2 = dx * dx + dy * dy
        mean_r2 = (cov_xx + cov_yy).view(b, 1, 1).clamp(min=1e-4)
        core_kernel = torch.exp(-r2 / (0.35 * mean_r2))
        ring_kernel = torch.exp(-((r2 - 1.0 * mean_r2) ** 2) / (0.5 * mean_r2 * mean_r2))
        core_prob = (probs * core_kernel).sum(dim=(1, 2)) / (core_kernel.sum(dim=(1, 2)) + 1e-4)
        ring_prob = (probs * ring_kernel).sum(dim=(1, 2)) / (ring_kernel.sum(dim=(1, 2)) + 1e-4)
        ring_hollowness = (ring_prob - core_prob).unsqueeze(1)  # (B, 1)

        gx = probs[:, :, 1:] - probs[:, :, :-1]
        gy = probs[:, 1:, :] - probs[:, :-1, :]
        perim = gx.abs().sum(dim=(1, 2)) + gy.abs().sum(dim=(1, 2)) + 1.0
        area_px = probs.sum(dim=(1, 2))
        compactness = (4.0 * 3.14159265 * area_px / (perim * perim)).clamp(0.0, 1.0).unsqueeze(1)  # (B, 1)

        return torch.cat([area_frac, log_elong, ring_hollowness, compactness], dim=1).to(seg_logits.dtype)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        has_cropmarks = x.shape[1] >= (NUM_TERRAIN_CHANNELS + 2)
        x_lidar = x[:, :NUM_TERRAIN_CHANNELS]
        s0 = self.full_skip(x_lidar)
        s1 = self.stem(x_lidar)
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

        if has_cropmarks:
            x_crop = x[:, NUM_TERRAIN_CHANNELS : NUM_TERRAIN_CHANNELS + 2]
            # Gate spatial cropmark residual by the soft LiDAR candidate probability + halo
            # so cropmarks sharpen/boost genuine low-relief LiDAR earthworks without creating
            # false-positive pixel peaks on flat tractor tramlines or field boundaries.
            lidar_prob = torch.sigmoid(seg_logits.detach())
            lidar_halo = F.max_pool2d(lidar_prob, kernel_size=7, stride=1, padding=3)
            lidar_gate = (0.10 + 0.90 * lidar_halo).clamp(0.0, 1.0)
            crop_in = torch.cat([d0.detach(), x_lidar[:, :4].detach(), x_crop], dim=1)
            crop_seg_out = torch.tanh(self.cropmark_seg_head(crop_in)) * 1.5
            seg_logits = seg_logits + lidar_gate * crop_seg_out[:, 0:1]
            cls_seg_logits = cls_seg_logits + lidar_gate * crop_seg_out[:, 1:]

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

        geom_feats = self._compute_d4_geometric_moments(seg_logits)
        geom_hidden = self.geom_mlp(geom_feats)

        tile_logits = 0.55 * self.tile_presence_out(hidden) + 0.45 * seg_q2 + self.geom_tile_out(geom_hidden)
        type_logits = self.tile_type_out(hidden) + 0.35 * cls_peaks + self.geom_type_out(geom_hidden)

        if has_cropmarks:
            x_crop = x[:, NUM_TERRAIN_CHANNELS : NUM_TERRAIN_CHANNELS + 2]
            fg_mask = torch.sigmoid(seg_logits.detach()).clamp(0.0, 1.0)
            fg_sum = fg_mask.sum(dim=(2, 3)).clamp(min=1.0)
            fg_exg = (x_crop[:, 0:1] * fg_mask).sum(dim=(2, 3)) / fg_sum
            fg_lum = (x_crop[:, 1:2] * fg_mask).sum(dim=(2, 3)) / fg_sum
            fg_exg_abs = (x_crop[:, 0:1].abs() * fg_mask).sum(dim=(2, 3)) / fg_sum
            fg_lum_abs = (x_crop[:, 1:2].abs() * fg_mask).sum(dim=(2, 3)) / fg_sum
            flat_exg = x_crop[:, 0].reshape(b, -1)
            flat_lum = x_crop[:, 1].reshape(b, -1)
            q_exg_pos = torch.topk(flat_exg, k=min(64, flat_exg.shape[1]), dim=1)[0].mean(dim=1, keepdim=True)
            q_exg_neg = torch.topk(-flat_exg, k=min(64, flat_exg.shape[1]), dim=1)[0].mean(dim=1, keepdim=True)
            q_lum_pos = torch.topk(flat_lum, k=min(64, flat_lum.shape[1]), dim=1)[0].mean(dim=1, keepdim=True)
            q_lum_neg = torch.topk(-flat_lum, k=min(64, flat_lum.shape[1]), dim=1)[0].mean(dim=1, keepdim=True)
            std_exg = flat_exg.std(dim=1, keepdim=True)
            std_lum = flat_lum.std(dim=1, keepdim=True)
            crop_vec = torch.cat(
                [
                    fg_exg,
                    fg_lum,
                    fg_exg_abs,
                    fg_lum_abs,
                    q_exg_pos,
                    q_exg_neg,
                    q_lum_pos,
                    q_lum_neg,
                    std_exg,
                    std_lum,
                ],
                dim=1,
            )
            crop_hidden = self.cropmark_mlp(crop_vec)
            tile_logits = tile_logits + self.cropmark_tile_out(crop_hidden)
            type_logits = type_logits + self.cropmark_type_out(crop_hidden)

        return {
            "seg_logits": seg_logits,
            "cls_seg_logits": cls_seg_logits,
            "tile_logits": tile_logits,
            "type_logits": type_logits,
            "seg_peak_logit": seg_q2,
            "geom_features": geom_feats,
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

        # 3. Multiple-Instance Learning (MIL) Peak & Multi-Scale Extent Suppression Loss:
        #    (a) Top-64 spatial peak logit must match target_tile (0 on negative tiles, 1 on positive tiles)
        #    (b) Top-2048 broad-area logit on negative tiles is suppressed to prevent whole-tile river/moorland blobs
        #    (c) Over-segmentation flood-fill penalty when predicted foreground area exceeds 2.2x ground-truth area
        flat_seg = seg_logits.view(seg_logits.shape[0], -1)
        topk_64 = torch.topk(flat_seg, k=min(64, flat_seg.shape[1]), dim=1)[0].mean(dim=1)
        ptw = torch.tensor(self.pos_tile_weight, device=seg_logits.device, dtype=seg_logits.dtype)
        mil_peak_loss = F.binary_cross_entropy_with_logits(topk_64, target_tile.float(), pos_weight=ptw)

        neg_tile_mask = target_tile < 0.5
        if torch.any(neg_tile_mask):
            topk_2048_neg = torch.topk(flat_seg[neg_tile_mask], k=min(2048, flat_seg.shape[1]), dim=1)[0].mean(dim=1)
            zeros_neg = torch.zeros_like(topk_2048_neg)
            mil_extent_loss = F.binary_cross_entropy_with_logits(topk_2048_neg, zeros_neg)
        else:
            mil_extent_loss = probs.mean() * 0.0

        pred_area = probs_valid.sum(dim=(1, 2))
        gt_area = bin_mask.sum(dim=(1, 2))
        valid_area = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
        overseg_spill = F.relu(pred_area - (2.2 * gt_area + 256.0)) / valid_area
        overseg_loss = overseg_spill.mean()

        # 4. Multi-class segmentation cross-entropy with ignore_index=255
        cls_w = self.cls_weights.to(device=cls_seg_logits.device, dtype=cls_seg_logits.dtype)
        ce_seg = F.cross_entropy(cls_seg_logits, target_mask.long(), weight=cls_w, ignore_index=255)

        # 5. Tile-level binary presence loss with prevalence correction
        tile_bce = F.binary_cross_entropy_with_logits(tile_logits, target_tile.float(), pos_weight=ptw)

        # 6. Tile-level multi-class type loss
        tw = self.type_weights.to(device=type_logits.device, dtype=type_logits.dtype)
        tile_ce = F.cross_entropy(type_logits, target_type.long(), weight=tw)

        total = (
            1.0 * focal_loss
            + 0.8 * dice_loss
            + 0.7 * mil_peak_loss
            + 0.4 * mil_extent_loss
            + 0.5 * overseg_loss
            + 0.5 * ce_seg
            + 0.8 * tile_bce
            + 0.4 * tile_ce
        )
        metrics = {
            "loss": float(total.detach().item()),
            "focal": float(focal_loss.detach().item()),
            "dice": float(dice_loss.detach().item()),
            "mil_peak": float(mil_peak_loss.detach().item()),
            "mil_extent": float(mil_extent_loss.detach().item()),
            "overseg": float(overseg_loss.detach().item()),
            "tile_bce": float(tile_bce.detach().item()),
        }
        return total, metrics

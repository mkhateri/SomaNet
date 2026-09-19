import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import math
import warnings


# The DINOv2 variants 
VARIANT = 'dinov2_vits14' # 'dinov2_vitb14','dinov2_vitl14', 'dinov2_vitg14'


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with GroupNorm.
    """

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides the channel count
        ng_1 = min(num_groups, in_ch)
        while in_ch % ng_1 != 0:
            ng_1 -= 1
        ng_2 = min(num_groups, out_ch)
        while out_ch % ng_2 != 0:
            ng_2 -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(ng_2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(ng_2, out_ch)
        self.act = nn.GELU()

        # Shortcut projection if channels change
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + identity)


class SpatialPriorModule(nn.Module):
    """
    Lightweight CNN branch that extracts local spatial features from the raw image.
    Provides fine-grained structure that ViT's patch embedding discards.
    #   Chen et al., "Vision Transformer Adapter for Dense Predictions",
    #   ICLR 2023. https://arxiv.org/abs/2205.08534

    """

    def __init__(self, in_ch: int = 3, stem_dim: int = 64, out_dim: int = 768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_dim), nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim), nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim), nn.GELU(),
        )
        self.out_proj = nn.Conv2d(stem_dim, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.stem(x))  # (B, out_dim, H/8, W/8)


class BiDirectionalAdapterBlock(nn.Module):
    """
    Two-way cross-attention between ViT patch tokens and spatial prior tokens.
    # Chen et al., "Vision Transformer Adapter for Dense Predictions",
    # ICLR 2023. https://arxiv.org/abs/2205.08534

    """

    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.0):
        super().__init__()

        # spatial <- patch
        self.norm_s_q = nn.LayerNorm(dim)
        self.norm_s_kv = nn.LayerNorm(dim)
        self.spatial_from_patch = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True)
        self.spatial_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

        # patch <- spatial
        self.norm_p_q = nn.LayerNorm(dim)
        self.norm_p_kv = nn.LayerNorm(dim)
        self.patch_from_spatial = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True)
        self.patch_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

        self.drop = nn.Dropout(drop)

    def forward(self, patch_tokens: torch.Tensor,
                spatial_tokens: torch.Tensor):
        """Returns updated (patch_tokens, spatial_tokens)."""
        # Step 1: spatial <- patch
        s_upd, _ = self.spatial_from_patch(
            self.norm_s_q(spatial_tokens),
            self.norm_s_kv(patch_tokens),
            self.norm_s_kv(patch_tokens),
        )
        spatial_tokens = spatial_tokens + self.drop(s_upd)
        spatial_tokens = spatial_tokens + self.drop(self.spatial_mlp(spatial_tokens))

        # Step 2: patch <- spatial
        p_upd, _ = self.patch_from_spatial(
            self.norm_p_q(patch_tokens),
            self.norm_p_kv(spatial_tokens),
            self.norm_p_kv(spatial_tokens),
        )
        patch_tokens = patch_tokens + self.drop(p_upd)
        patch_tokens = patch_tokens + self.drop(self.patch_mlp(patch_tokens))

        return patch_tokens, spatial_tokens


class MultiScaleFusionHead(nn.Module):
    """
    #  Xie et al., "SegFormer: Simple and Efficient Design for Semantic
    #  Segmentation with Transformers", NeurIPS 2021.

    """

    def __init__(self, hidden_dim: int, embed_dim: int, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales

        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, embed_dim),
            )
            for _ in range(num_scales)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(num_scales * embed_dim, embed_dim, 1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
        )

    def forward(self, scale_tokens: list, ph: int, pw: int,
                out_size: tuple) -> torch.Tensor:
        """
        Args:
            scale_tokens: list of N tensors, each (B, ph*pw, hidden_dim)
            ph, pw: patch grid size
            out_size: (H, W) target resolution

        Returns: (B, embed_dim, H, W)
        """
        scale_feats = []
        for i, tokens in enumerate(scale_tokens):
            B = tokens.shape[0]
            feat = self.scale_projs[i](tokens)                    # (B, N, embed_dim)
            feat = feat.permute(0, 2, 1).reshape(B, -1, ph, pw)  # (B, embed_dim, ph, pw)
            feat = F.interpolate(feat, size=out_size,
                                 mode='bilinear', align_corners=False)
            scale_feats.append(feat)

        fused = torch.cat(scale_feats, dim=1)  # (B, N*embed_dim, H, W)
        return self.fuse(fused)                 # (B, embed_dim, H, W)


class FullResolutionSkip(nn.Module):
    """
    Shallow CNN branch that processes the raw input at full resolution.
    """

    def __init__(self, in_ch: int = 1, skip_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, skip_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_dim),
            nn.GELU(),
            nn.Conv2d(skip_dim, skip_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_dim),
            nn.GELU(),
            nn.Conv2d(skip_dim, skip_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeepResidualDecoder(nn.Module):
    """
    Deep residual decoder with skip-connection structure and full-resolution
    skip from the input image.
    """

    def __init__(self, embed_dim: int = 64, mid_dim: int = 128, skip_dim: int = 32):
        super().__init__()
        self.has_skip = skip_dim > 0
        in_dim = embed_dim + skip_dim if skip_dim > 0 else embed_dim
        self.block1 = ResidualConvBlock(in_dim, mid_dim)
        self.block2 = ResidualConvBlock(mid_dim, mid_dim)
        self.block3 = ResidualConvBlock(mid_dim, embed_dim)
        self.block4 = ResidualConvBlock(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                full_res_skip: torch.Tensor = None) -> torch.Tensor:
        if full_res_skip is not None:
            identity = x                       # skip from the fusion input (embed_dim only)
            x = torch.cat([x, full_res_skip], dim=1)  # (B, embed_dim + skip_dim, H, W)
        else:
            identity = x
        x = self.block1(x)
        skip1 = x                              # skip from block 1
        x = self.block2(x) + skip1             # mid-level skip
        x = self.block3(x)
        x = self.block4(x) + identity          # input-level skip (embed_dim)
        return x


class EmbeddingRefinementHead(nn.Module):
    """
    Multi-layer embedding head that refines features before affinity computation.
    """

    def __init__(self, embed_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        ng_embed = min(8, embed_dim)
        while embed_dim % ng_embed != 0:
            ng_embed -= 1
        ng_emb = min(8, emb_dim)
        while emb_dim % ng_emb != 0:
            ng_emb -= 1

        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(ng_embed, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, emb_dim, 3, padding=1, bias=False),
            nn.GroupNorm(ng_emb, emb_dim),
            nn.GELU(),
            nn.Conv2d(emb_dim, emb_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x)


class SegmentationHead(nn.Module):

    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.seg = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
            nn.Conv2d(emb_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg(x)


# ==================
# Main model
# ==================

class DINOv2Seg(nn.Module):
    """
    Self-contained DINOv2 segmentation model with deep decoder.
    """

    _CONFIGS = {
        'dinov2_vits14': {'hidden_dim': 384,  'patch_size': 14},
        'dinov2_vitb14': {'hidden_dim': 768,  'patch_size': 14},
        'dinov2_vitl14': {'hidden_dim': 1024, 'patch_size': 14},
        'dinov2_vitg14': {'hidden_dim': 1536, 'patch_size': 14},
    }

    def __init__(
        self,
        model_name: str = VARIANT,
        in_channels: int = 1,
        num_classes: int = 2,
        emb_dim: int = 32,
        embed_dim: int = 64,
        decoder_mid_dim: int = 128,
        adapter_depth: int = 4,
        use_checkpoint: bool = False,
        pretrained: bool = True,
        img_size: int = 400,         # for SwinIR config compat (not used internally)
        in_chans: int = None,        # alias for in_channels (SwinIR compat)
    ):
        super().__init__()

        if in_chans is not None:
            in_channels = in_chans

        # Variant used in this work
        if model_name == 'auto':
            model_name = VARIANT

        assert model_name in self._CONFIGS, \
            f"model_name must be one of {list(self._CONFIGS.keys())} or 'auto'"

        cfg = self._CONFIGS[model_name]
        hidden_dim = cfg['hidden_dim']
        self.patch_size = cfg['patch_size']
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        self.model_name = model_name

        # Store output dims for external use
        self.embed_dim = embed_dim
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        # Input projection (grayscale -> RGB) 
        self.in_channels = in_channels
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.input_proj.weight, 1.0 / in_channels)
        else:
            self.input_proj = None

        #==============================================================
        #   Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
        #   TMLR 2024. https://arxiv.org/abs/2304.07193
        #==============================================================
        # DINOv2 backbone 
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=pretrained
        )
        self.backbone_blocks = self.backbone.blocks  

        # Spatial prior module 
        self.spatial_prior = SpatialPriorModule(
            in_ch=3, stem_dim=64, out_dim=hidden_dim
        )

        # Learnable positional encoding for spatial tokens 
        self.register_parameter("sp_pos_embed", None)
        self._sp_pos_shape = None

        # Adapter + multi-scale extraction indices 
        n_blocks = len(self.backbone_blocks)
        step = max(1, n_blocks // adapter_depth)
        self.adapter_indices = [
            min((i + 1) * step - 1, n_blocks - 1) for i in range(adapter_depth)
        ]

        # Bidirectional adapter blocks 
        num_heads = max(1, hidden_dim // 64)  
        self.adapter_blocks = nn.ModuleList([
            BiDirectionalAdapterBlock(hidden_dim, num_heads)
            for _ in range(adapter_depth)
        ])

        # Multi-scale feature fusion 
        self.fpn_head = MultiScaleFusionHead(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_scales=adapter_depth,
        )

        # Feature normalization 
        self.feat_norm = nn.GroupNorm(1, embed_dim)

        # Full-resolution skip branch 
        self.skip_dim = 32
        self.full_res_skip = FullResolutionSkip(
            in_ch=in_channels,
            skip_dim=self.skip_dim,
        )

        # Deep residual decoder 
        self.decoder = DeepResidualDecoder(
            embed_dim=embed_dim,
            mid_dim=decoder_mid_dim,
            skip_dim=self.skip_dim,
        )

        # Embedding refinement head 
        self.embedding_head = EmbeddingRefinementHead(
            embed_dim=embed_dim,
            emb_dim=emb_dim,
        )

        # Segmentation head 
        self.seg_head = SegmentationHead(emb_dim, num_classes)

        # Initialize non-backbone weights
        self._init_weights()

    # Weight initialization
    def _init_weights(self):
        """Initialize adapter, decoder, and head weights. Backbone is pretrained."""
        modules_to_init = [
            self.spatial_prior, *self.adapter_blocks,
            self.fpn_head, self.feat_norm, self.full_res_skip,
            self.decoder, self.embedding_head, self.seg_head,
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # Backbone freezing for 2-stage training
    def freeze_backbone(self, n_unfrozen: int = 0):
        """
        Freeze DINOv2 backbone for efficient training.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if n_unfrozen > 0:
            for blk in self.backbone_blocks[-n_unfrozen:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        print(f"[DINOv2Seg] Backbone: {frozen}/{total} params frozen "
              f"(last {n_unfrozen} blocks unfrozen)")

    def get_param_groups(self, lr_backbone: float = 1e-5,
                         lr_new: float = 5e-4) -> list:
        """
        Get parameter groups with different learning rates for optimizer.
        """
        backbone_params = []
        new_params = []

        backbone_param_ids = set(id(p) for p in self.backbone.parameters())

        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_param_ids:
                backbone_params.append(p)
            else:
                new_params.append(p)

        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': lr_backbone})
        if new_params:
            groups.append({'params': new_params, 'lr': lr_new})

        return groups

    # Spatial positional encoding 
    def _get_sp_pos_embed(self, ph: int, pw: int,
                          device: torch.device) -> torch.Tensor:
        if self.sp_pos_embed is None or self._sp_pos_shape != (ph, pw):
            pos = nn.Parameter(
                torch.zeros(1, ph * pw, self.hidden_dim, device=device)
            )
            nn.init.trunc_normal_(pos, std=0.02)
            if self.sp_pos_embed is None:
                self.register_parameter("sp_pos_embed", pos)
            else:
                self.sp_pos_embed = pos
            self._sp_pos_shape = (ph, pw)
        return self.sp_pos_embed

    # Forward through DINOv2 + adapter
    def _forward_with_adapter(self, x_pad: torch.Tensor,
                               sp_tokens: torch.Tensor) -> list:

        x_tok = self.backbone.prepare_tokens_with_masks(x_pad, masks=None)

        scale_tokens = []
        adapter_idx = 0

        for block_i, block in enumerate(self.backbone_blocks):
            if self.use_checkpoint and block.training:
                x_tok = grad_checkpoint(block, x_tok, use_reentrant=False)
            else:
                x_tok = block(x_tok)

            if (adapter_idx < len(self.adapter_indices) and
                    block_i == self.adapter_indices[adapter_idx]):

                patch_tokens = x_tok[:, 1:]  

                # Bidirectional adapter
                patch_tokens, sp_tokens = self.adapter_blocks[adapter_idx](
                    patch_tokens, sp_tokens
                )

                # Collect for multi-scale fusion
                scale_tokens.append(patch_tokens)

                # Write back into ViT stream
                x_tok = torch.cat([x_tok[:, :1], patch_tokens], dim=1)
                adapter_idx += 1

        return scale_tokens

    # Forward pass
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Full-resolution skip 
        # Processes raw input at pixel level to preserve fine details
        fr_skip = self.full_res_skip(x) 

        # Input projection 
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Pad to multiple of patch_size 
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        ph = H_pad // self.patch_size
        pw = W_pad // self.patch_size

        # Spatial prior
        sp_feat = self.spatial_prior(x_pad)
        sp_feat = F.interpolate(sp_feat, size=(ph, pw),
                                mode='bilinear', align_corners=False)
        sp_tokens = sp_feat.flatten(2).transpose(1, 2)  
        sp_tokens = sp_tokens + self._get_sp_pos_embed(ph, pw, x.device)

        # DINOv2 + bidirectional adapter
        scale_tokens = self._forward_with_adapter(x_pad, sp_tokens)

        # Multi-scale feature fusion 
        feat = self.fpn_head(scale_tokens, ph, pw, out_size=(H, W))
        feat = self.feat_norm(feat)

        # Deep residual decoder with full-res skip
        feat = self.decoder(feat, full_res_skip=fr_skip)

        # Embedding refinement 
        x_emb = self.embedding_head(feat)

        # Segmentation head 
        x_seg = self.seg_head(x_emb)

        return x_emb, x_seg

#===============================================================
if __name__ == '__main__':
    print("="*60)
    print("DINOv2Seg — Self-Contained Model Test")
    print("="*60)

    model = DINOv2Seg(
        model_name='dinov2_vits14',   # use vits14 for testing
        in_channels=1,
        num_classes=2,
        emb_dim=32,
        embed_dim=64,
        pretrained=False,             # skip download for testing
    )

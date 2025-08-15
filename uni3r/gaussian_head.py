import torch
import torch.nn as nn
from einops import rearrange
from simple_knn._C import distCUDA2

from .utils.gaussian_model import build_covariance
from .utils.sh_utils import RGB2SH


class GaussianHead(nn.Module):

    def __init__(self, d_pt_feat=64, **kwargs):
        super().__init__()
        # args
        self.args = kwargs
        self.d_means = 3
        self.d_scales = 3
        self.d_rotations = 4
        self.d_opacities = 1
        self.sh_degree = 3
        self.d_view_dep_features = 3  # RGB
        self.d_sh = (self.sh_degree + 1)**2
        self.d_attr = (
            self.d_scales + self.d_rotations + self.d_opacities +
            self.d_view_dep_features * self.d_sh)
        if self.args.get('d_gs_feats'):
            self.d_attr += self.args['d_gs_feats']

        # Create a mask for the spherical harmonics coefficients.
        # This ensures that at initialization, the coefficients are biased
        # towards having a large DC component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh, ), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1)**2] = 0.5 * 0.25**degree

        self.gaussian_proj = nn.Linear(d_pt_feat, self.d_attr)

        # Activation functions
        self.scale_activation = torch.exp
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid

    def forward(self, point_transformer_output, lseg_res_feature):
        if 'world_points' not in point_transformer_output:  # PointTransformer
            scene_scale = point_transformer_output['scale']  # B, 1, 1
            scene_center = point_transformer_output['center']  # B, 1, 3
            B, H, W, _ = point_transformer_output['shape']
            V = 2
            normalized_means = point_transformer_output['coord']  # B * V * H * W, 3
            colors = point_transformer_output['color']  # B * V * H * W, 3

            # split normalized_means to 2 views
            normalized_means = rearrange(
                normalized_means, '(b v h w) c -> v b (h w) c', v=2, b=B, h=H, w=W)
            means = normalized_means * scene_scale + scene_center  # V, B, H * W, 3
            means = rearrange(means, 'v b (h w) c -> b (v h w) c', b=B, v=2, h=H, w=W)

            # get features
            feat = point_transformer_output['feat']
            gaussian_attr = self.gaussian_proj(feat)
        else:  # VGGT
            means = point_transformer_output['world_points'].flatten(1, 3)
            gaussian_attr = point_transformer_output['gs_attrs']
            B, V, _, H, W = gaussian_attr.shape
            gaussian_attr = gaussian_attr.permute(0, 1, 3, 4, 2).flatten(0, 3)

        scales, rotations, opacities, sh_coeffs, gs_feats, colors = torch.split(
            gaussian_attr, [
                self.d_scales, self.d_rotations, self.d_opacities,
                self.d_view_dep_features * self.d_sh, self.args['d_gs_feats'],
                3
            ],
            dim=-1)

        # scales
        # calculate the distance between each point and its nearest neighbor
        all_dist = torch.stack([
            torch.sqrt(torch.clamp_min(distCUDA2(pts3d), 0.0000001))
            for pts3d in means
        ])  # B, V * H * W
        median_dist = all_dist.median(dim=-1)[0][:, None, None]  # B, 1, 1
        scales = scales.clamp_max(10.0)
        scales = self.scale_activation(scales) * 0.0025
        scales = rearrange(scales, '(b v h w) c -> b (v h w) c', b=B, v=V, h=H, w=W)
        scales = scales * all_dist[..., None]
        # clip scales
        scales = torch.clamp(scales, min=0.1 * median_dist, max=3.0 * median_dist)
        scales = rearrange(scales, 'b (v h w) c -> (b v h w) c', b=B, v=V, h=H, w=W)

        # activation
        rotations = self.rotation_activation(rotations)
        opacities = self.opacity_activation(opacities)

        # build covariance matrix
        covs = build_covariance(scales, rotations)

        # sh_mask
        sh_coeffs = rearrange(
            sh_coeffs, '(b v h w) (c d) -> (b v h w) c d', 
            b=B, v=V, h=H, w=W, c=self.d_sh, d=self.d_view_dep_features)
        sh_dc = sh_coeffs[..., 0, :]
        sh_rest = sh_coeffs[..., 1:, :]
        if self.args.get('rgb_residual'):
            # denormalize colors
            colors = colors * 0.5 + 0.5
            sh_rgb = RGB2SH(colors)  # (B * V * H * W, 3)
            # add rgb residual to dc component
            sh_dc = sh_dc + sh_rgb
            # concatenate dc and rest
            sh_coeffs = torch.cat([sh_dc[..., None, :], sh_rest], dim=-2)
        sh_coeffs = sh_coeffs * self.sh_mask[None, :, None]

        # lseg_features(learning residual)
        lseg_res_feature = rearrange(
            lseg_res_feature, '(v b) c h w -> (b v h w) c', b=B, v=V, h=H, w=W)
        gs_feats = gs_feats + lseg_res_feature

        # split to 2 views
        scales = rearrange(scales, '(b v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)
        rotations = rearrange(rotations, '(b v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)
        opacities = rearrange(opacities, '(b v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)
        sh_coeffs = rearrange(sh_coeffs, '(b v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)
        covs = rearrange(covs, '(b v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)
        means = rearrange(means, 'b (v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)
        gs_feats = rearrange(gs_feats, '(b v h w) ... -> v b h w ...', v=V, b=B, h=H, w=W)

        preds = []
        for i in range(scales.shape[0]):
            pred = {}
            pred['scales'] = scales[i]
            pred['rotations'] = rotations[i]
            pred['covs'] = covs[i]
            pred['opacities'] = opacities[i]
            pred['sh_coeffs'] = sh_coeffs[i]
            pred['means'] = means[i]
            pred['gs_feats'] = gs_feats[i]
            preds.append(pred)
        return preds

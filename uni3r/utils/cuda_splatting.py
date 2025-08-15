#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gsplat import rasterization

from .gaussian_model import GaussianModel
from .sh_utils import eval_sh
from .graphics_utils import getWorld2View2, getProjectionMatrix


class DummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy

class DummyPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

def calculate_fov(output_width, output_height, focal_length, aspect_ratio=1.0, invert_y=False):
    fovx = 2 * math.atan((output_width / (2 * focal_length)))
    fovy = 2 * math.atan((output_height / aspect_ratio) / (2 * focal_length))

    if invert_y:
        fovy = -fovy

    return fovx, fovy

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           intrinsics=None, extrinsics=None, scale=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if 'GaussianRasterizer' in globals():
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python or 'GaussianRasterizer' not in globals():
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    semantic_feature = pc.get_semantic_feature

    near = 0.1
    far = 100.0
    if scale is not None:
        near = near / scale
        far = far / scale
        scale = 1.0 / near # 0.1, 1
        extrinsics = extrinsics.clone()
        extrinsics[:3, 3] = extrinsics[:3, 3] * scale
        scales = scales * scale
        rotations = rotations * scale
        means3D = means3D * scale
        near = near * scale
        far = far * scale

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if 'GaussianRasterizer' in globals():
        rendered_image, feature_map, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            semantic_feature = semantic_feature,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered, _, meta = rasterization(
            means3D,
            rotations,
            scales,
            opacity.squeeze(1),
            torch.cat([semantic_feature.squeeze(1), colors_precomp], dim=-1),
            extrinsics[None],
            intrinsics[None],
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            near_plane=near,
            far_plane=far,
            render_mode='RGB+D',
            channel_chunk=32)

        feature_map, rendered_image, depth = torch.split(
            rendered, [semantic_feature.size(-1), 3, 1], dim=-1)
        radii = meta['radii']

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'feature_map': feature_map,
            "depth": depth}

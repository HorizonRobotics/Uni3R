import torch
from einops import rearrange

# merge points from two views and add color information
def merge_points(dust3r_output, view1, view2, grid_size=0.01):
    # get points from dust3r_output
    points1 = dust3r_output[0]['pts3d'].detach() # B, H, W, 3
    points2 = dust3r_output[1]['pts3d_in_other_view'].detach() # B, H, W, 3
    shape = points1.shape
    # add color information
    colors = torch.stack([view1['img'], view2['img']], dim=1) # B, V, 3, H, W
    colors = rearrange(colors, 'b v c h w -> b (v h w) c') # B, V * H * W, 3
    # merge points
    points = torch.stack([points1, points2], dim=1) # B, V, H, W, 3
    points = rearrange(points, 'b v h w c -> b (v h w) c') # B, V * H * W, 3
    B, N, _ = points.shape
    offset = torch.arange(1, B + 1, device=points.device) * N
    # Center and normalize points
    center = torch.mean(points, dim=1, keepdim=True)  # compute centroid
    points = points - center  # center the points
    # Normalize points using coordinate range
    max_coords = torch.max(torch.abs(points), dim=1, keepdim=True)[0]  # find max absolute value for each dimension
    scale = torch.max(max_coords, dim=2, keepdim=True)[0]  # get the overall scale factor
    points = points / scale  # normalize points to fit in a unit cube
    # concat points and colors
    feat = torch.cat([points, colors], dim=-1) # B, V * H * W, 6
    
    data_dict = {
        'coord': rearrange(points, 'b n c -> (b n) c'),
        'color': rearrange(colors, 'b n c -> (b n) c'),
        'feat': rearrange(feat, 'b n c -> (b n) c'),
        'offset': offset,
        'grid_size': grid_size,
        'center': center,
        'scale': scale,
        'shape': shape,
    }

    return data_dict

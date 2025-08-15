import sys
import os
import numpy as np
import scipy.interpolate
import PIL
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2

from dust3r.utils.image import heif_support_enabled, exif_transpose, _resize_pil_image, ImgNorm
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from .cuda_splatting import render, DummyPipeline
from .gaussian_model import GaussianModel
from .camera_utils import get_scaled_camera
from ..loss import merge_and_split_predictions
from .camera_utils import move_c2w_along_z

from einops import rearrange
LABELS = ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']
NUM_LABELS = len(LABELS) + 1
PALLETE = plt.cm.get_cmap('tab10', NUM_LABELS)
COLORS_LIST = [PALLETE(i)[:3] for i in range(NUM_LABELS)]
COLORS = torch.tensor(COLORS_LIST, dtype=torch.float32)

def load_images(folder_or_list, size, square_ok=False, verbose=True, save_dir=None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if (size == 224) or (size == 256):
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if (size == 224) or (size == 256):
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//32)*16, ((2*cy)//32)*16
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        
        # Save the processed image if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"processed_{len(imgs):03d}.png")
            img.save(save_path)
            if verbose:
                print(f' - saved processed image to {save_path}')
        
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def poses_to_points(poses, dist):
    """Converts from pose matrices to (position, lookat, up) format."""
    pos = poses[:, :3, -1]
    lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
    up = poses[:, :3, -1] + dist * poses[:, :3, 1]
    return np.stack([pos, lookat, up], 1)

def points_to_poses(points):
    """Converts from (position, lookat, up) format to pose matrices."""
    return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

def interp(points, n, k, s):
    """Runs multidimensional B-spline interpolation on the input points."""
    sh = points.shape
    pts = np.reshape(points, (sh[0], -1))
    k = min(k, sh[0] - 1)
    tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
    u = np.linspace(0, 1, n, endpoint=False)
    new_points = np.array(scipy.interpolate.splev(u, tck))
    new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
    return new_points

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points) 

def batch_visualize_tensor_global_pca(tensor_batch, num_components=3):
    B, C, H, W = tensor_batch.shape

    tensor_flat_all = tensor_batch.reshape(B, C, -1).permute(1, 0, 2).reshape(C, -1).T

    tensor_flat_all_np = tensor_flat_all.cpu().numpy()

    scaler = StandardScaler()
    tensor_flat_all_np = scaler.fit_transform(tensor_flat_all_np)

    pca = PCA(n_components=num_components)
    tensor_reduced_all_np = pca.fit_transform(tensor_flat_all_np)

    tensor_reduced_all = torch.tensor(tensor_reduced_all_np, dtype=tensor_batch.dtype).T.reshape(num_components, B, H * W).permute(1, 0, 2)

    output_tensor = torch.zeros((B, 3, H, W))

    for i in range(B):
        tensor_reduced = tensor_reduced_all[i].reshape(num_components, H, W)
        tensor_reduced -= tensor_reduced.min()
        tensor_reduced /= tensor_reduced.max()
        output_tensor[i] = tensor_reduced[:3]

    return output_tensor

def depth_to_colormap(depth_tensor, colormap='jet'):
    B, _, _, _ = depth_tensor.shape

    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())

    depth_np = depth_tensor.squeeze(1).cpu().numpy()

    cmap = plt.get_cmap(colormap)
    colored_images = []

    for i in range(B):
        colored_image = cmap(depth_np[i])
        colored_images.append(colored_image[..., :3])

    colored_tensor = torch.tensor(np.array(colored_images), dtype=torch.float32).permute(0, 3, 1, 2)
    
    return colored_tensor

def save_video(frames, video_path, fps=24):
    """Save video using OpenCV
    
    Args:
        frames: List of frames in numpy array format
        video_path: Output video path
        fps: Frames per second
    
    Raises:
        ValueError: If frames list is empty
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")
        
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    try:
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    finally:
        # Ensure proper cleanup of video writer
        out.release()

def tensors_to_videos(all_images, all_depth_vis, all_fmap_vis, all_sems_vis, video_dir='videos', fps=24):
    B, C, H, W = all_images.shape
    assert all_depth_vis.shape == (B, C, H, W)
    assert all_fmap_vis.shape == (B, C, H, W)
    assert all_sems_vis.shape == (B, C, H, W)
    os.makedirs(video_dir, exist_ok=True)

    all_images = (all_images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    all_depth_vis = (all_depth_vis.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    all_fmap_vis = (all_fmap_vis.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    all_sems_vis = (all_sems_vis.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    save_video(all_images, os.path.join(video_dir, 'output_images_video.mp4'), fps=fps)
    save_video(all_depth_vis, os.path.join(video_dir, 'output_depth_video.mp4'), fps=fps)
    save_video(all_fmap_vis, os.path.join(video_dir, 'output_fmap_video.mp4'), fps=fps)
    save_video(all_sems_vis, os.path.join(video_dir, 'output_sems_video.mp4'), fps=fps)

    print(f'Videos saved to {video_dir}')

def transfer_images_to_device(images, device):
    """
    Transfer the loaded images to the specified device.
    
    Args:
        images (list): List of dictionaries containing image data.
        device (str or torch.device): The device to transfer the data to.
    
    Returns:
        list: List of dictionaries with image data transferred to the specified device.
    """
    transferred_images = []
    for img_dict in images:
        transferred_dict = {
            'img': img_dict['img'].to(device),
            'true_shape': torch.tensor(img_dict['true_shape'], device=device),
            'idx': img_dict['idx'],
            'instance': img_dict['instance']
        }
        transferred_images.append(transferred_dict)
    return transferred_images

def render_camera_path(video_poses, camera_params, gaussians, model, device, pipeline, bg_color, image_shape):
    """Helper function to render camera path
    
    Args:
        video_poses: List of camera poses
        camera_params: Camera parameters containing extrinsics and intrinsics
        gaussians: Gaussian model
        model: Feature extraction model
        device: Computation device
        pipeline: Rendering pipeline
        bg_color: Background color
        image_shape: Image dimensions
    
    Returns:
        rendered_images: Rendered images
        rendered_feats: Rendered feature maps
        rendered_depths: Rendered depth maps
        rendered_sems: Rendered semantic maps
    """
    extrinsics, intrinsics = camera_params
    rendered_images = []
    rendered_feats = []
    rendered_depths = []
    rendered_sems = []
    
    for i in range(len(video_poses)):
        target_extrinsics = torch.zeros(4, 4).to(device)
        target_extrinsics[3, 3] = 1.0
        target_extrinsics[:3, :4] = torch.tensor(video_poses[i], device=device)
        camera = get_scaled_camera(extrinsics[0], target_extrinsics, intrinsics[0], 1.0, image_shape)
        
        rendered_output = render(camera, gaussians, pipeline, bg_color)
        rendered_images.append(rendered_output['render'])
        
        # Process feature map
        feature_map = rendered_output['feature_map']
        feature_map = model.feature_expansion(feature_map[None, ...])
        
        # Process semantic map
        logits = model.lseg_feature_extractor.decode_feature(feature_map, labelset=LABELS)
        semantic_map = torch.argmax(logits, dim=1) + 1
        mask = COLORS[semantic_map.cpu()]
        mask = rearrange(mask, 'b h w c -> b c h w')
        rendered_sems.append(mask.squeeze(0))
            
        # Downsample and upsample feature map
        feature_map = feature_map[:, ::16, ...]
        feature_map = torch.nn.functional.interpolate(feature_map, scale_factor=2, mode='bilinear', align_corners=True)
        rendered_feats.append(feature_map[0])
        del feature_map
        
        rendered_depths.append(rendered_output['depth'])

    # Stack and process results
    rendered_images = torch.clamp(torch.stack(rendered_images, dim=0), 0, 1)
    rendered_feats = torch.stack(rendered_feats, dim=0)
    rendered_depths = torch.stack(rendered_depths, dim=0)
    rendered_sems = torch.stack(rendered_sems, dim=0)
        
    return rendered_images, rendered_feats, rendered_depths, rendered_sems

@torch.no_grad()
def render_video_from_file(file_list, model, output_path, device='cuda', resolution=224, n_interp=90, fps=30, path_type='default'):
    # 1. Load images
    images = load_images(file_list, resolution, save_dir=os.path.join(output_path, 'processed_images'))
    images = transfer_images_to_device(images, device)  # Transfer images to the specified device
    image_shape = images[0]['true_shape'][0]
    
    # 2. Get camera pose    
    pairs = make_pairs(images, prefilter=None, symmetrize=True)
    output = inference(pairs, model.dust3r.dust3r, device, batch_size=1)
    mode = GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    extrinsics = scene.get_im_poses()
    intrinsics = scene.get_intrinsics()
    video_poses = generate_interpolated_path(extrinsics[:, :3, :].cpu().numpy(), n_interp=n_interp)
    
    # 3. Get gaussians
    pred1, pred2 = model(*images)
    pred = merge_and_split_predictions(pred1, pred2)
    gaussians = GaussianModel.from_predictions(pred[0], sh_degree=3)
    
    # 4. Render original viewpoint
    pipeline = DummyPipeline()
    bg_color = torch.tensor([0.0, 0.0, 0.0]).to(device)
    camera_params = (extrinsics, intrinsics)
    
    rendered_images, rendered_feats, rendered_depths, rendered_sems = render_camera_path(
        video_poses, camera_params, gaussians, model, device, pipeline, bg_color, image_shape)
    
    # 5. Visualization
    all_fmap_vis = batch_visualize_tensor_global_pca(rendered_feats)
    all_depth_vis = depth_to_colormap(rendered_depths)
    all_sems_vis = rendered_sems
    
    # 6. Save videos and gaussian point cloud
    tensors_to_videos(rendered_images, all_depth_vis, all_fmap_vis, all_sems_vis, output_path, fps=fps)
    gaussians.save_ply(os.path.join(output_path, 'gaussians.ply'))
    
    # 7. Render moved viewpoint
    moved_extrinsics = move_c2w_along_z(extrinsics, 2.0)
    moved_video_poses = generate_interpolated_path(moved_extrinsics[:, :3, :].cpu().numpy(), n_interp=n_interp)
    camera_params = (extrinsics, intrinsics)
    
    moved_rendered_images, moved_rendered_feats, moved_rendered_depths, moved_rendered_sems = render_camera_path(
        moved_video_poses, camera_params, gaussians, model, device, pipeline, bg_color, image_shape)
    
    # 8. Visualize and save moved results
    moved_all_fmap_vis = batch_visualize_tensor_global_pca(moved_rendered_feats)
    moved_all_depth_vis = depth_to_colormap(moved_rendered_depths)
    moved_all_sems_vis = moved_rendered_sems
    
    moved_output_path = os.path.join(output_path, 'moved')
    os.makedirs(moved_output_path, exist_ok=True)
    tensors_to_videos(moved_rendered_images, moved_all_depth_vis, moved_all_fmap_vis, moved_all_sems_vis, 
                     moved_output_path, fps=fps)

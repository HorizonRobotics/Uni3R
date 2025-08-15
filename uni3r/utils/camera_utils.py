import math
import torch
from dust3r.utils.geometry import inv
from .cuda_splatting import DummyCamera

def get_scaled_camera(ref_camera_extrinsics, target_camera_extrinsics, target_camera_intrinsics, scale, image_shape):
    """
    get a scaled camera from a reference camera to a target camera
    
    """
    
    # get extrinsics(target_camera to ref_camera)
    if ref_camera_extrinsics is not None:
        target_camera_extrinsics = inv(ref_camera_extrinsics) @ target_camera_extrinsics
    # scale translation
    target_camera_extrinsics[:3, 3] = target_camera_extrinsics[:3, 3] * scale
    # invert extrinsics(ref_camera to target_camera)
    target_camera_extrinsics_inv = inv(target_camera_extrinsics)
    # calculate fov
    fovx = 2 * math.atan(image_shape[1] / (2 * target_camera_intrinsics[0, 0]))
    fovy = 2 * math.atan(image_shape[0] / (2 * target_camera_intrinsics[1, 1]))
    # return camera(numpy)
    R = target_camera_extrinsics_inv[:3, :3].cpu().numpy().transpose() # R.transpose() : ref_camera_2_target_camera
    T = target_camera_extrinsics_inv[:3, 3].cpu().numpy() # T : ref_camera_2_target_camera
    image_shape = image_shape.cpu().numpy()
    return DummyCamera(R, T, fovx, fovy, image_shape[1], image_shape[0])

def move_c2w_along_z(extrinsics: torch.Tensor, distance: float) -> torch.Tensor:
    """
    Move multiple Camera-to-World (C2W) matrices backward, making cameras move away from origin along their respective Z axes.

    Args:
        extrinsics (torch.Tensor): Tensor of shape [N, 4, 4] containing N C2W matrices.
        distance (float): Distance to move backward.

    Returns:
        torch.Tensor: Updated C2W matrices with same shape as input.
    """
    # Ensure input is a 4D matrix with last dimension being 4x4
    assert extrinsics.dim() == 3 and extrinsics.shape[1:] == (4, 4), \
        "Input extrinsics must be a tensor of shape [N, 4, 4]"

    # Create a copy to avoid modifying the original matrix
    updated_extrinsics = extrinsics.clone()

    # Iterate through each C2W matrix
    for i in range(updated_extrinsics.shape[0]):
        # Extract rotation matrix R and translation vector t
        R = updated_extrinsics[i, :3, :3]  # shape [3, 3]
        t = updated_extrinsics[i, :3, 3]   # shape [3]

        # Get camera's Z axis direction (third column)
        z_axis = R[:, 2]  # shape [3]

        # Calculate new translation vector, moving backward along Z axis
        t_new = t - distance * z_axis

        # Update translation part of C2W matrix
        updated_extrinsics[i, :3, 3] = t_new

    return updated_extrinsics

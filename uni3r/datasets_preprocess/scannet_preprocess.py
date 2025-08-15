import os
import numpy as np
import cv2
import torch
import torch.multiprocessing as mp

def process_scene_on_gpu(gpu_id, scene_names, data_root, output_queue):
    torch.cuda.set_device(gpu_id)
    local_pairs = {}
    local_images = {}

    for scene_name in scene_names:
        save_path = os.path.join(data_root, scene_name, "scene_data.npz")
        if os.path.exists(save_path):
            print(f"Scene {scene_name} already processed, skipping")
            continue
        pairs, images = process_scene(data_root, scene_name)
        np.savez_compressed(save_path, pairs=pairs, images=images)

    output_queue.put((local_pairs, local_images))

def preprocess_scannet(data_root, threads_per_gpu=4):
    scene_names = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
    num_gpus = torch.cuda.device_count()
    total_threads = num_gpus * threads_per_gpu
    
    # Evenly distribute scenes across all threads
    scenes_per_thread = [scene_names[i::total_threads] for i in range(total_threads)]
    
    output_queue = mp.Queue()
    processes = []
    
    # Create multiple processes for each GPU
    for gpu_id in range(num_gpus):
        for thread_id in range(threads_per_gpu):
            process_id = gpu_id * threads_per_gpu + thread_id
            p = mp.Process(
                target=process_scene_on_gpu, 
                args=(gpu_id, scenes_per_thread[process_id], data_root, output_queue)
            )
            p.start()
            processes.append(p)

    # Collect results from all processes
    all_pairs = {}
    all_images = {}
    for _ in range(total_threads):
        local_pairs, local_images = output_queue.get()
        all_pairs.update(local_pairs)
        all_images.update(local_images)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Save to npz file
    np.savez_compressed(os.path.join(data_root, "scannet_image_pairs.npz"), **all_pairs)
    np.savez_compressed(os.path.join(data_root, "scannet_images.npz"), **all_images)

    # print the number of image pairs
    # sum up the number of image pairs for all scenes
    total_pairs = sum(len(pairs) for pairs in all_pairs.values())
    print(f"Total number of image pairs: {total_pairs}")
    return all_pairs, all_images

def process_scene(data_root, scene_name):
    pairs = []
    images_dir = os.path.join(data_root, scene_name, "images")
    images = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if file.endswith(".jpg")]
    images.sort()

    # Check validity of c2w for each image
    valid_images = []
    for image in images:
        _, c2w, _ = load_image(data_root, scene_name, image)
        if is_valid_c2w(c2w):
            valid_images.append(image)
        else:
            print(f"Invalid c2w for image {image} in scene {scene_name}")

    # generate image pairs
    slide_window = 50
    num_sub_intervals = 5
    
    pairs = generate_image_pairs(data_root, scene_name, valid_images, slide_window, num_sub_intervals)
    print(f"Scene {scene_name} has {len(pairs)} image pairs and {len(valid_images)} valid images out of {len(images)} total images")
    return pairs, valid_images

def is_valid_c2w(c2w):
    return not np.any(np.isinf(c2w)) and not np.any(np.isnan(c2w))

def generate_image_pairs(data_root, scene_name, images, slide_window, num_sub_intervals=3):
    pairs = []
    n = len(images)
    
    # Define IOU sub-intervals
    iou_range = (0.3, 0.8)
    sub_interval_size = (iou_range[1] - iou_range[0]) / num_sub_intervals
    sub_intervals = [(iou_range[0] + i * sub_interval_size, iou_range[0] + (i + 1) * sub_interval_size) 
                     for i in range(num_sub_intervals)]
    
    for i in range(n):
        # Keep track of whether a pair has been added for each sub-interval
        interval_selected = [False] * num_sub_intervals
        
        for j in range(i+1, min(i + slide_window, n)):
            # Break early if all sub-intervals have been selected
            if all(interval_selected):
                break
            
            # Load image pair
            depth1, c2w1, K1 = load_image(data_root, scene_name, images[i])
            depth2, c2w2, K2 = load_image(data_root, scene_name, images[j])
            
            # Calculate mean IoU
            try:
                iou_1 = calculate_iou(depth1, c2w1, K1, depth2, c2w2, K2)
                iou_2 = calculate_iou(depth2, c2w2, K2, depth1, c2w1, K1)
            except Exception as e:
                print(f"Error calculating IoU for images {images[i]} and {images[j]} in scene {scene_name}: {str(e)}")
                continue
            
            mean_iou = (iou_1 + iou_2) / 2
            
            # Check which sub-interval the mean IoU falls into
            for idx, (lower, upper) in enumerate(sub_intervals):
                if lower <= mean_iou <= upper and not interval_selected[idx]:
                    pairs.append((i, j, mean_iou))
                    interval_selected[idx] = True  # Mark this interval as selected
                    break  # Move to the next pair after adding one in the current sub-interval

    return pairs


def load_image(data_root, scene_name, image_id):
    # load depthmap
    depth_path = f"{data_root}/{scene_name}/depths/{image_id}.png"
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    # load camera parameters
    meta_path = f"{data_root}/{scene_name}/images/{image_id}.npz"
    meta = np.load(meta_path)
    c2w = meta['camera_pose']
    K = meta['camera_intrinsics']
    return depth, c2w, K

# Unproject depthmap to point cloud and project to another camera
def calculate_iou(depth1, c2w1, K1, depth2, c2w2, K2):
    # Move data to GPU and ensure float32 dtype
    depth1 = torch.from_numpy(depth1).cuda().float()
    depth2 = torch.from_numpy(depth2).cuda().float()
    c2w1 = torch.from_numpy(c2w1).cuda().float()
    c2w2 = torch.from_numpy(c2w2).cuda().float()
    K1 = torch.from_numpy(K1).cuda().float()
    K2 = torch.from_numpy(K2).cuda().float()

    # Get image dimensions
    h, w = depth1.shape

    # Create pixel coordinates
    y, x = torch.meshgrid(torch.arange(h, device='cuda', dtype=torch.float32),
                            torch.arange(w, device='cuda', dtype=torch.float32))
    pixels = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x.flatten())), dim=-1).T

    # Unproject pixels to 3D points
    pixels_3d = torch.linalg.inv(K1) @ pixels
    pixels_3d *= depth1.flatten().unsqueeze(0)

    # Transform 3D points to world coordinates
    pixels_world = c2w1[:3, :3] @ pixels_3d + c2w1[:3, 3:4]

    # Check if c2w2[:3, :3] is invertible
    if torch.det(c2w2[:3, :3]) == 0:
        return 0, False  # Calculation failed

    # Project world points to second camera
    pixels_cam2 = torch.linalg.inv(c2w2[:3, :3]) @ (pixels_world - c2w2[:3, 3:4])
    pixels_img2 = K2 @ pixels_cam2

    # Normalize homogeneous coordinates
    pixels_img2 = pixels_img2[:2] / pixels_img2[2]
    pixels_img2 = pixels_img2.T

    # Filter valid pixels
    valid_mask = (pixels_img2[:, 0] >= 0) & (pixels_img2[:, 0] < w) & \
                    (pixels_img2[:, 1] >= 0) & (pixels_img2[:, 1] < h)
    
    pixels_img2 = pixels_img2[valid_mask].long()

    # Compare depths
    projected_depth = pixels_cam2[2, valid_mask]
    actual_depth = depth2[pixels_img2[:, 1], pixels_img2[:, 0]]

    depth_diff = torch.abs(projected_depth - actual_depth)
    depth_threshold = 0.1  # 10cm threshold

    overlap_mask = depth_diff < depth_threshold

    # Calculate IoU
    intersection = torch.sum(overlap_mask)
    union = torch.sum(valid_mask) + torch.sum(depth2 > 0) - intersection

    iou = intersection.float() / union.float() if union > 0 else torch.tensor(0.0, device='cuda')

    return iou.item()

if __name__ == "__main__":
    data_root = "data/scannet_processed"
    # Number of threads per GPU can be specified through parameters
    preprocess_scannet(data_root, threads_per_gpu=12)

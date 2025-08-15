import os
import os.path as osp
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import numpy as np
import cv2
from dust3r.utils.image import imread_cv2
import itertools


def camera_normalization(pivotal_pose: np.ndarray, poses: np.ndarray):
    canonical_camera_extrinsics = np.eye(4, dtype=np.float32)
    pivotal_pose_inv = np.linalg.inv(pivotal_pose)
    camera_norm_matrix = canonical_camera_extrinsics @ pivotal_pose_inv
    poses = camera_norm_matrix @ poses
    return poses

def segment_uniform_sample(frame_num, num_segments):
    if frame_num < num_segments:
        raise ValueError(f"帧数不足:frame_num={frame_num} < num_segments={num_segments}")

    segment_size = frame_num / num_segments
    scene_combinations = []

    for j in range(int(np.round(segment_size))):
        indices = []
        for i in range(num_segments):
            start = int(np.round(i * segment_size))
            end = int(np.round((i + 1) * segment_size)) - 1
            end = min(end, frame_num - 1)
            index = start + j
            indices.append(index)
        scene_combinations.append(tuple(indices))

    return scene_combinations

class Scannet(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, num_views=2, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self._load_data()

    def _load_data(self):
        # Traverse all the folders in the data_root
        scene_names = [folder for folder in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, folder))]
        scene_names.sort()
        if self.split == 'train':
            scene_names = scene_names[:-150]
        else:
            scene_names = scene_names[-150:]
        # merge all pairs and images
        pairs = [] # (scene_name, image_idx1, image_idx2)
        images = {} # scene_name -> list of image_paths
        for scene_name in scene_names:
            images_paths = os.listdir(osp.join(self.ROOT, scene_name, 'color'))
            images_paths.sort()
            frame_num = len(images_paths)
            if frame_num > 32:
                if self.num_views == 2:
                    scene_combinations = [(i, j)
                                        for i, j in itertools.combinations(range(frame_num), 2)
                                        if 0 < abs(i - j) <= 4 and abs(i - j) % 2 == 0]
                elif self.num_views == 4:
                    scene_combinations = [(i, j)
                                        for i, j in itertools.combinations(range(frame_num), 2)
                                        if 0 < abs(i - j) <= 12 and abs(i - j) % 6 == 0]
                elif self.num_views == 8:
                    scene_combinations = [(i, j)
                                        for i, j in itertools.combinations(range(frame_num), 2)
                                        if 0 < abs(i - j) <= 28 and abs(i - j) % 14 == 0]
                elif self.num_views == 16:
                    # scene_combinations = segment_uniform_sample(frame_num=128, num_segments=self.num_views)
                    scene_combinations = [(i, j)
                                        for i, j in itertools.combinations(range(frame_num), 2)
                                        if 0 < abs(i - j) <= 60 and abs(i - j) % 30 == 0]
                elif self.num_views == 32:
                    # scene_combinations = segment_uniform_sample(frame_num=128, num_segments=self.num_views)
                    scene_combinations = [(i, j)
                                        for i, j in itertools.combinations(range(frame_num), 2)
                                        if 0 < abs(i - j) <= 124 and abs(i - j) % 62 == 0]
                # TODO: other nums of views input
                else:
                    print(f"num_views is wrong!!!")
                    assert False

                pairs.extend([(scene_name, *pair) for pair in scene_combinations])
                images[scene_name] = images_paths
            
        self.pairs = pairs
        self.images = images
        
    def __len__(self):
        return len(self.pairs)
    
    def _get_views(self, idx, resolution, rng):

        if self.num_views==8 or 16 or 32:

            total_nums = self.num_views
            scene_name, image_idx1, image_idx2 = self.pairs[idx]
            image_idx1 = int(image_idx1)
            image_idx2 = int(image_idx2)
            idx_gap = int(abs(image_idx2 - image_idx1) / (total_nums - 1))

            pick_id = []
            # context views here
            for i in range(self.num_views):
                pick_id.append(image_idx1 + i * idx_gap)

            target_id = [(pick_id[i] + pick_id[i+1]) // 2 for i in range(len(pick_id)-1)]
            pick_id += target_id

            views = []

            for i, view_idx in enumerate(pick_id):
                basename = os.path.basename(self.images[scene_name][view_idx]).split('.')[0]
                # Load RGB image
                rgb_path = osp.join(self.ROOT, scene_name, 'color', f'{basename}.png')
                # rgb_image = imread_cv2(rgb_path)
                rgb_image = imread_cv2(rgb_path).copy()
                # Load depthmap
                depthmap_path = osp.join(self.ROOT, scene_name, 'depth', f'{basename}.png')
                # depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
                depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED).copy()
                depthmap = depthmap.astype(np.float32) / 1000
                depthmap[~np.isfinite(depthmap)] = 0  # invalid
                # Load camera parameters
                meta_path = osp.join(self.ROOT, scene_name, 'pose', f'{basename}.npz')
                with np.load(meta_path) as meta:
                    intrinsics = meta['camera_intrinsics']
                    camera_pose = meta['camera_pose']
                # meta = np.load(meta_path)
                # intrinsics = meta['camera_intrinsics']
                # camera_pose = meta['camera_pose']
                # crop if necessary
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
                
                extrinsics = np.copy(camera_pose)
                scale = np.float32(1.0)
                # store all informations
                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    extrinsics=extrinsics.astype(np.float32),
                    scale=np.float32(scale),
                    dataset='ScanNet',
                    label=scene_name + '_' + basename,
                    instance=f'{str(idx)}_{str(view_idx)}',
                ))

            if self.num_views == 2:
                scale = np.linalg.norm(views[0]['extrinsics'][:3, 3] - views[-1]['extrinsics'][:3, 3])

            for i, view_idx in enumerate(pick_id):
                views[i]['extrinsics'][:3, 3] /=  scale
                if i == 0:
                    extrinsics0 = views[i]['extrinsics']
                views[i]['extrinsics'] = camera_normalization(extrinsics0, views[i]['extrinsics'])
                views[i]['scale'] = scale

            return views
           
        else:
            pick_id = []

            for i in range(self.num_views):
                pick_id.append(self.pairs[idx][i+1])
            scene_name = self.pairs[idx][0]

            target_id = [int((pick_id[0] + pick_id[-1])/2)]
            pick_id += target_id

            views = []

            for i, view_idx in enumerate(pick_id):
                basename = os.path.basename(self.images[scene_name][view_idx]).split('.')[0]
                # Load RGB image
                rgb_path = osp.join(self.ROOT, scene_name, 'color', f'{basename}.png')
                rgb_image = imread_cv2(rgb_path)
                # Load depthmap
                depthmap_path = osp.join(self.ROOT, scene_name, 'depth', f'{basename}.png')
                depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
                depthmap = depthmap.astype(np.float32) / 1000
                depthmap[~np.isfinite(depthmap)] = 0  # invalid
                # Load camera parameters
                meta_path = osp.join(self.ROOT, scene_name, 'pose', f'{basename}.npz')
                meta = np.load(meta_path)
                intrinsics = meta['camera_intrinsics']
                camera_pose = meta['camera_pose']
                # crop if necessary
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
                
                extrinsics = np.copy(camera_pose)
                scale = np.float32(1.0)
                # store all informations
                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    extrinsics=extrinsics.astype(np.float32),
                    scale=np.float32(scale),
                    dataset='ScanNet',
                    label=scene_name + '_' + basename,
                    instance=f'{str(idx)}_{str(view_idx)}',
                ))

            for i, view_idx in enumerate(pick_id):
                views[i]['extrinsics'][:3, 3] /=  scale
                if i == 0:
                    extrinsics0 = views[i]['extrinsics']
                views[i]['extrinsics'] = camera_normalization(extrinsics0, views[i]['extrinsics'])
                views[i]['scale'] = scale

            return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = Scannet(split='train', ROOT="data/scannet_processed", resolution=224, aug_crop=16)

    print(len(dataset))

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 3
        print(view_name(views[0]), view_name(views[1]), view_name(views[2]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1, 2]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1, 2]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
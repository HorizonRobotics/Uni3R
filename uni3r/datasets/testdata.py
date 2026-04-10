import os.path as osp
import json
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from collections import deque, namedtuple
import numpy as np
import cv2
import os
import struct
from dust3r.utils.image import imread_cv2
import pandas as pd
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

from .scannet import camera_normalization

# COLMAP data structures
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file"""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """Read COLMAP cameras binary file"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[camera_properties[1]].num_params
            params = read_next_bytes(fid, 8*num_params, "d"*num_params)
            cameras[camera_id] = Camera(
                id=camera_properties[0],
                model=camera_properties[1],
                width=camera_properties[2],
                height=camera_properties[3],
                params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    """Read COLMAP images binary file"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, 24*num_points2D, "ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                  tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = BaseImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def map_func(label_path, labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
    labels = [label.lower() for label in labels]

    df = pd.read_csv(label_path, sep='\t')
    id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()

    nyu40class_to_newid = {cls: labels.index(cls) + 1 if cls in labels else labels.index('other') + 1 for cls in set(id_to_nyu40class.values())}

    id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}

    return np.vectorize(lambda x: id_to_newid.get(x, labels.index('other') + 1) if x != 0 else 0)


class TestDataset(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, llff_hold=8, test_ids=[1,4], is_training=False, num_views=2, normalize_camera=True, *args, ROOT, **kwargs):
        
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.num_views = num_views
        self.map_func = map_func(os.path.join(ROOT, 'scannetv2-labels.combined.tsv'))
        
        if self.num_views == 2:
            # load all scenes for multiviews evaluation
            with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
                self.scenes = json.load(f)
                self.scenes = {k: sorted(v) for k, v in self.scenes.items() if len(v) > 0}
                ignored_scenes = ['scene0696_02']
                for key in ignored_scenes:
                    if key in self.scenes:
                        del self.scenes[key]
        # load all scenes for 360 evaluation
        elif self.num_views in (4, 8, 16, 32):
            self.scenes = {}
            # Traverse all the folders in the data_root
            scene_names = [folder for folder in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, folder))]
            scene_names.sort()
            for scene_name in scene_names:
                images_paths = [f for f in os.listdir(osp.join(self.ROOT, scene_name, 'images'))
                                if f.lower().endswith('.jpg')]
                images_paths.sort()
                indices = np.arange(0, self.num_views*2, 2).astype(int)
                # indices = indices + 15
                context_views = [images_paths[i][:-4] for i in indices]
                filter_images = images_paths[:self.num_views*2-1]
                target_views_with_extension = [f for f in filter_images if f[:-4] not in context_views]
                target_views = [f[:-4] for f in target_views_with_extension]
                self.scenes[scene_name] = context_views + target_views
                ignored_scenes = ['scene0696_02']
                for key in ignored_scenes:
                    if key in self.scenes:
                        del self.scenes[key]
        else:
            print(f"num_views is wrong!!!")
            assert False
        
        self.scene_list = list(self.scenes.keys())
        self.invalidate = {scene: {} for scene in self.scene_list}
        
        self.llff_hold = llff_hold
        self.test_ids = test_ids
        self.is_training = is_training
        self.all_views = self.get_all_views()
        self.views_per_scene = len(self.all_views) // len(self.scene_list)

    def __len__(self):
        return len(self.all_views)
    
    def get_all_views(self):
        views = []
        for scene_id in self.scene_list:
            if not self.is_training:
                if self.num_views == 2:
                    selected_views = [i for i in range(len(self.scenes[scene_id])) if i % self.llff_hold in self.test_ids]
                    for target_view in selected_views:
                        source_view1 = max(target_view - 1, 0)
                        source_view2 = min(target_view + 1, len(self.scenes[scene_id]) - 1)
                        views.append((scene_id, (target_view, source_view2, source_view1)))
                elif self.num_views == 4 or 8 or 16 or 32:
                    selected_views = list(range(self.num_views*2-1))[::-1]
                    views.append((scene_id, selected_views))
            else:
                selected_views = [i for i in range(len(self.scenes[scene_id])) if i % self.llff_hold not in self.test_ids]
                for target_view in selected_views:
                    source_view1 = target_view
                    source_view2 = target_view + 1 if target_view + 1 < len(self.scenes[scene_id]) else target_view - 1
                    views.append((scene_id, (target_view, source_view2, source_view1)))

        return views
    
    def _get_views(self, idx, resolution, rng):
        # choose a scene
        scene_id, imgs_idxs = self.all_views[idx]

        image_pool = self.scenes[scene_id]

        if resolution not in self.invalidate[scene_id]:  # flag invalid images
            self.invalidate[scene_id][resolution] = [False for _ in range(len(image_pool))]
            
        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))
        
        views = []
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()
            
            if self.invalidate[scene_id][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[scene_id][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break
        
            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, scene_id, 'images', f'{view_idx}.jpg')
            meta_data_path = impath.replace('jpg', 'npz')
            depthmap_path = impath.replace('images', 'depths').replace('.jpg', '.png')
            labelmap_path = impath.replace('images', 'labels').replace('.jpg', '.png')

            # load camera params
            input_metadata = np.load(meta_data_path)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            has_inf = np.isinf(camera_pose)
            contains_inf = np.any(has_inf)
            if contains_inf:
                self.invalidate[scene_id][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
            
            # load image and depth and mask
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
            maskmap = np.ones_like(depthmap) * 255 # don't use mask for now
            labelmap = imread_cv2(labelmap_path, cv2.IMREAD_UNCHANGED)
            # pack
            depth_mask_map = np.stack([depthmap, maskmap, labelmap], axis=-1)
                
            # crop if necessary
            rgb_image, depth_mask_map, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depth_mask_map, intrinsics, resolution, rng=rng, info=impath)
            # unpack
            depthmap = depth_mask_map[:, :, 0]
            maskmap = depth_mask_map[:, :, 1]
            labelmap = depth_mask_map[:, :, 2]
            # map labelmap
            labelmap = self.map_func(labelmap)
            
            depthmap = (depthmap.astype(np.float32) / 1000)
            if mask_bg:
                # load object mask
                maskmap = maskmap.astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[scene_id][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            extrinsics = np.copy(camera_pose)
            scale = np.float32(1.0)

            view = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                extrinsics=extrinsics,
                scale=np.float32(scale),
                dataset='Testdata',
                label=scene_id,
                instance=osp.split(impath)[1],
                labelmap=labelmap,
            )
            views.append(view)
        
        if self.num_views == 2:
            scale = np.linalg.norm(views[0]['extrinsics'][:3, 3] - views[1]['extrinsics'][:3, 3])
        else:
            scale = np.linalg.norm(views[0]['extrinsics'][:3, 3] - views[self.num_views-1]['extrinsics'][:3, 3])

        for i in range(len(views)):
            views[i]['extrinsics'][:3, 3] /=  scale
            if i == 0:
                extrinsics0 = views[i]['extrinsics']
            views[i]['extrinsics'] = camera_normalization(extrinsics0, views[i]['extrinsics'])
            views[i]['scale'] = scale
            
        return views
    
    def get_test_views(self, scene_id, view_idx, resolution):
        if type(resolution) == int:
            resolution = (resolution, resolution)
        else:
            resolution = tuple(resolution)
            
        impath = osp.join(self.ROOT, scene_id, 'images', f'{view_idx}.jpg')
        meta_data_path = impath.replace('jpg', 'npz')
        depthmap_path = impath.replace('images', 'depths').replace('.jpg', '.png')
        labelmap_path = impath.replace('images', 'labels').replace('.jpg', '.png')
        
        # load camera params
        input_metadata = np.load(meta_data_path)
        camera_pose = input_metadata['camera_pose'].astype(np.float32)
        intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
        
        # if camera_pose has NaNs, return None
        if not np.isfinite(camera_pose).all():
            return None
        
        # load image and depth and mask
        rgb_image = imread_cv2(impath)
        depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
        maskmap = np.ones_like(depthmap) * 255 # don't use mask for now
        labelmap = imread_cv2(labelmap_path, cv2.IMREAD_UNCHANGED)
        
        # pack
        depth_mask_map = np.stack([depthmap, maskmap, labelmap], axis=-1)
        
        # crop if necessary
        rgb_image, depth_mask_map, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depth_mask_map, intrinsics, resolution, rng=None, info=impath)
        
        # unpack
        depthmap = depth_mask_map[:, :, 0]
        maskmap = depth_mask_map[:, :, 1]
        labelmap = depth_mask_map[:, :, 2]
        
        # map labelmap
        labelmap = self.map_func(labelmap)
        
        depthmap = (depthmap.astype(np.float32) / 1000)
        # load object mask
        maskmap = maskmap.astype(np.float32)
        maskmap = (maskmap / 255.0) > 0.1

        # update the depthmap with mask
        depthmap *= maskmap
        
        view = dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose,
            labelmap=labelmap,
            camera_intrinsics=intrinsics,
            extrinsics=input_metadata['extrinsics'].astype(np.float32),
            dataset='Scannet',
            label=scene_id,
            instance=osp.split(impath)[1],
        )
        assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
        view['idx'] = (view_idx)

        # encode the image
        width, height = view['img'].size
        view['true_shape'] = np.int32((height, width))
        view['img'] = self.transform(view['img'])

        assert 'camera_intrinsics' in view
        if 'camera_pose' not in view:
            view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
        else:
            assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
        assert 'pts3d' not in view
        assert 'valid_mask' not in view
        assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

        view['pts3d'] = pts3d
        view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
        
        return view


class MipNeRF360Dataset(BaseStereoViewDataset):
    """Dataset loader for MipNeRF360 datasets in COLMAP format"""
    
    def __init__(self, mask_bg=True, llff_hold=8, test_ids=[2, 6], is_training=False, 
                 num_views=2, normalize_camera=True, *args, ROOT, **kwargs):
        
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.num_views = num_views
        self.normalize_camera = normalize_camera
        self.llff_hold = llff_hold
        self.test_ids = test_ids
        self.is_training = is_training
        
        # Load COLMAP data
        self.cameras, self.images = self._load_colmap_data()
        
        # Get all images
        image_list = list(self.images.keys())
        image_list.sort()
        self.image_list = image_list
        self.n_images = len(self.image_list)
        
        # Get all views
        self.all_views = self._get_all_views()
        self.invalidate = {}

        # Only for labelmap, No use for now
        self.map_func = map_func(os.path.join('/horizon-bucket/robot_lab/users/haoyi.jiang/data/scannet_test', 'scannetv2-labels.combined.tsv'))
        
    def _load_colmap_data(self):
        """Load COLMAP reconstruction files"""
        sparse_dir = osp.join(self.ROOT, 'sparse', '0')
        cameras = read_cameras_binary(osp.join(sparse_dir, 'cameras.bin'))
        images = read_images_binary(osp.join(sparse_dir, 'images.bin'))
        return cameras, images
    
    def _get_all_views(self):
        """Generate all view combinations based on split"""
        views = []
        
        if not self.is_training:
            if self.num_views == 2:
                # For testing with 2 views
                selected_indices = [i for i in range(self.n_images) if i % self.llff_hold in self.test_ids]
                for target_idx in selected_indices:
                    source_idx1 = max(target_idx - 2, 0)
                    source_idx2 = min(target_idx + 2, self.n_images - 1)
                    views.append((target_idx, source_idx2, source_idx1))
                print(views)
            elif self.num_views == 4:
                # For testing with 4 views
                selected_indices = [i for i in range(self.n_images) if i % self.llff_hold in self.test_ids]
                for target_idx in selected_indices:
                    source_idx1 = max(target_idx - 4, 0)
                    source_idx2 = max(target_idx - 2, 0)
                    source_idx3 = min(target_idx + 2, self.n_images - 1)
                    source_idx4 = min(target_idx + 4, self.n_images - 1)
                    target_idx1 = max(target_idx - 2, 0)
                    target_idx2 = target_idx
                    target_idx3 = min(target_idx + 2, self.n_images - 1)
                    views.append((target_idx2, source_idx4, source_idx3, source_idx2, source_idx1))
                print(views)
            elif self.num_views == 8:
                # For testing with 4 views
                selected_indices = [i for i in range(self.n_images) if i % self.llff_hold in self.test_ids]
                for target_idx in selected_indices:
                    source_idx1 = max(target_idx - 8, 0)
                    source_idx2 = max(target_idx - 6, 0)
                    source_idx3 = max(target_idx - 4, 0)
                    source_idx4 = max(target_idx - 2, 0)
                    source_idx5 = min(target_idx + 2, self.n_images - 1)
                    source_idx6 = min(target_idx + 4, self.n_images - 1)
                    source_idx7 = min(target_idx + 6, self.n_images - 1)
                    source_idx8 = min(target_idx + 8, self.n_images - 1)
                    target_idx1 = max(target_idx - 6, 0)
                    target_idx2 = max(target_idx - 4, 0)
                    target_idx3 = max(target_idx - 2, 0)
                    target_idx4 = target_idx
                    target_idx5 = min(target_idx + 2, self.n_images - 1)
                    target_idx6 = min(target_idx + 4, self.n_images - 1)
                    target_idx7 = min(target_idx + 6, self.n_images - 1)
                    views.append((target_idx4, source_idx8, source_idx7, source_idx6, source_idx5, source_idx4, source_idx3, source_idx2, source_idx1))
                print(views)
            # elif self.num_views == 16:
            #     selected_indices = [i for i in range(self.n_images) if i % self.llff_hold in self.test_ids]
            #     for target_idx in selected_indices:
            #         for i in range(self.n_images):
            #             source_idx = max(target_idx - i, 0)
            else:
                # For 360 evaluation with more views
                selected_indices = list(range(min(self.num_views * 2 - 1, self.n_images)))[::-1]
                views.append(selected_indices)
        else:
            # Training views (excluding test views)
            selected_indices = [i for i in range(self.n_images) if i % self.llff_hold not in self.test_ids]
            for target_idx in selected_indices:
                source_idx1 = target_idx
                source_idx2 = target_idx + 1 if target_idx + 1 < self.n_images else target_idx - 1
                views.append((target_idx, source_idx2, source_idx1))
        
        return views
    
    def __len__(self):
        return len(self.all_views)
    
    def _colmap_to_camera_pose(self, qvec, tvec):
        """Convert COLMAP qvec and tvec to 4x4 world-to-camera pose matrix"""
        R = qvec2rotmat(qvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = tvec
        return pose

    def _get_c2w(self, qvec, tvec):
        """Convert COLMAP qvec and tvec to 4x4 camera-to-world pose matrix"""
        T_w2c = self._colmap_to_camera_pose(qvec, tvec)
        # c2w = inv(w2c)
        T_c2w = np.linalg.inv(T_w2c)
        return T_c2w
    
    def _get_intrinsics(self, camera_id):
        """Get camera intrinsics from COLMAP camera"""
        camera = self.cameras[camera_id]
        model = camera.model
        
        # For PINHOLE model (model=1), params = [fx, fy, cx, cy]
        if model == 1:
            fx, fy, cx, cy = camera.params
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            # For other models, just use params
            if len(camera.params) >= 4:
                fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
            else:
                # Default intrinsic estimation
                fx = camera.params[0]
                fy = camera.params[0] if len(camera.params) < 2 else camera.params[1]
                cx, cy = camera.width / 2, camera.height / 2
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        
        return intrinsics
    
    def _get_views(self, idx, resolution, rng):
        """Get views for a given index"""
        view_indices = self.all_views[idx]
        
        # Decide if we mask the background
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))
        
        views = []
        imgs_idxs = deque(view_indices if isinstance(view_indices, (list, tuple)) else [view_indices])
        
        if resolution not in self.invalidate:
            self.invalidate[resolution] = [False] * self.n_images
        
        while len(imgs_idxs) > 0:
            if isinstance(imgs_idxs, deque):
                im_idx = imgs_idxs.pop()
            else:
                im_idx = imgs_idxs[0]
                imgs_idxs = imgs_idxs[1:]
            
            if self.invalidate[resolution][im_idx]:
                # Search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, self.n_images):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % self.n_images
                    if not self.invalidate[resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break
            
            # Get image info
            image_id = self.image_list[im_idx]
            image_info = self.images[image_id]
            image_name = image_info.name
            
            # Load image
            impath = osp.join(self.ROOT, 'images', image_name)
            rgb_image = imread_cv2(impath)
            from PIL import Image
            rgb_image = Image.fromarray(rgb_image)
            
            # Get camera pose and intrinsics
            camera_pose = self._get_c2w(image_info.qvec, image_info.tvec)
            
            # Check for invalid pose
            if np.any(np.isinf(camera_pose)) or np.any(np.isnan(camera_pose)):
                self.invalidate[resolution][im_idx] = True
                continue
            
            intrinsics = self._get_intrinsics(image_info.camera_id)
            
            # Get actual image dimensions
            width, height = rgb_image.size  # PIL Image size is (width, height)
            
            # Scale intrinsics to match actual image size
            # Get original camera params
            camera = self.cameras[image_info.camera_id]
            original_width = camera.width
            original_height = camera.height
            
            # Scale intrinsics if image was resized
            if width != original_width or height != original_height:
                scale_x = width / original_width
                scale_y = height / original_height
                intrinsics = intrinsics.copy()
                intrinsics[0, 0] *= scale_x  # fx
                intrinsics[1, 1] *= scale_y  # fy
                intrinsics[0, 2] *= scale_x  # cx
                intrinsics[1, 2] *= scale_y  # cy
            
            # Create dummy depthmap (placeholder)
            depthmap = np.ones((height, width), dtype=np.float32) * 10.0  # 10m default depth
            maskmap = np.ones((height, width), dtype=np.float32) * 255.0
            
            # Pack depth and mask
            depth_mask_map = np.stack([depthmap, maskmap], axis=-1)

            # Crop if necessary
            if hasattr(self, '_crop_resize_if_necessary'):
                try:
                    rgb_image, depth_mask_map, intrinsics = self._crop_resize_if_necessary(
                        rgb_image, depth_mask_map, intrinsics, resolution, rng=rng, info=impath)
                except Exception as e:
                    print(f"Warning: crop_resize failed for {image_name}: {e}")
                    # Fall back to simple resize
                    rgb_image = rgb_image.resize(resolution)
                    # Scale depth_mask_map - use cv2 for proper interpolation
                    # depth_mask_map shape is (height, width, 2)
                    depth_mask_map = cv2.resize(depth_mask_map, resolution, interpolation=cv2.INTER_NEAREST)
                    # Update intrinsics
                    scale_x = resolution[0] / width
                    scale_y = resolution[1] / height
                    intrinsics = intrinsics.copy()
                    intrinsics[0, 0] *= scale_x
                    intrinsics[1, 1] *= scale_y
                    intrinsics[0, 2] *= scale_x
                    intrinsics[1, 2] *= scale_y
            else:
                # Simple resize
                if resolution != (width, height):
                    rgb_image = rgb_image.resize(resolution)
                    # Scale depth_mask_map
                    depth_mask_map = cv2.resize(depth_mask_map, resolution, interpolation=cv2.INTER_NEAREST)
            
            # Unpack
            depthmap = depth_mask_map[:, :, 0]
            maskmap = depth_mask_map[:, :, 1]
            labelmap = depth_mask_map[:, :, 1]
            # map labelmap
            labelmap = self.map_func(labelmap)
            
            # Apply mask if needed
            depthmap = depthmap.astype(np.float32)
            if mask_bg:
                maskmap = maskmap.astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap *= maskmap
            
            # Check for valid depth
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                self.invalidate[resolution][im_idx] = True
                continue
            
            extrinsics = np.copy(camera_pose)
            scale = np.float32(1.0)
            
            view = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                extrinsics=extrinsics,
                scale=scale,
                dataset='MipNeRF360',
                label=osp.basename(self.ROOT),
                instance=image_name,
                labelmap=labelmap,
            )
            views.append(view)
        
        # Normalize cameras
        scale = np.linalg.norm(views[0]['extrinsics'][:3, 3] - views[self.num_views - 1]['extrinsics'][:3, 3])
        
        for i in range(len(views)):
            views[i]['extrinsics'][:3, 3] /= scale
            if i == 0:
                extrinsics0 = views[i]['extrinsics']
            views[i]['extrinsics'] = camera_normalization(extrinsics0, views[i]['extrinsics'])
            views[i]['scale'] = scale
        
        return views
    
if __name__ == "__main__":
    # from dust3r.datasets.base.base_stereo_view_dataset import view_name
    # from dust3r.viz import SceneViz, auto_cam_size
    # from dust3r.utils.image import rgb

    # Test MipNeRF360 dataset
    mipnerf_dataset = MipNeRF360Dataset(
        split='test',
        ROOT="/home/users/xiangyu.sun/dataset/mipnerf360/bicycle",
        resolution=(512, 512),
        num_views=2,
        is_training=False
    )
    print(f"MipNeRF360 dataset length: {len(mipnerf_dataset)}")
    print(f"Number of images: {mipnerf_dataset.n_images}")
    
    if len(mipnerf_dataset) > 0:
        views = mipnerf_dataset[0]
        print(f"Number of views in sample: {len(views)}")
        if len(views) > 0:
            print(f"View keys: {views[0].keys()}")
            print(f"Image size: {views[0]['img'].size}")
            print(f"Camera intrinsics:\n{views[0]['camera_intrinsics']}")
            print(f"Camera pose:\n{views[0]['camera_pose']}")
    
    # Original test data
    # dataset = TestDataset(split='test', ROOT="data/scannet_test", resolution=(512, 384))
    # print(len(dataset))
    # for idx in np.random.permutation(len(dataset)):
    #     views = dataset[idx]
    #     # assert len(views) == 2
    #     print(view_name(views[0]), view_name(views[-1]))
    #     viz = SceneViz()
    #     poses = [views[view_idx]['camera_pose'] for view_idx in [0, -1]]
    #     cam_size = max(auto_cam_size(poses), 0.001)
    #     for view_idx in [0, 1]:
    #         img = views[view_idx]['img']
    #         pts3d = views[view_idx]['pts3d']
    #         # save pts3d to file
    #         pts3d_path = f'{view_idx}_scannetpp_pts3d.ply'
    #         # save_pcd(pts3d, img.permute(1, 2, 0).numpy(), pts3d_path)
    #         valid_mask = views[view_idx]['valid_mask']
    #         colors = rgb(views[view_idx]['img'])
    #         viz.add_pointcloud(pts3d, colors, valid_mask)
    #         viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
    #                        focal=views[view_idx]['camera_intrinsics'][0, 0],
    #                        color=(idx*255, (1 - idx)*255, 0),
    #                        image=colors,
    #                        cam_size=cam_size)
    #     viz.show()
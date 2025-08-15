import os.path as osp
import json
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from collections import deque
import numpy as np
import cv2
import os
from dust3r.utils.image import imread_cv2
import pandas as pd
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

from .scannet import camera_normalization


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
        elif self.num_views == 4 or 8 or 16 or 32:
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
                elif self.num_views == 8 or 16 or 32:
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
    
if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = TestDataset(split='test', ROOT="data/scannet_test", resolution=(512, 384))
    print(len(dataset))
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        print(view_name(views[0]), view_name(views[-1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, -1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            img = views[view_idx]['img']
            pts3d = views[view_idx]['pts3d']
            # save pts3d to file
            pts3d_path = f'{view_idx}_scannetpp_pts3d.ply'
            # save_pcd(pts3d, img.permute(1, 2, 0).numpy(), pts3d_path)
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
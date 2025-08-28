import lpips
import torch
import torch.nn.functional as F
from dust3r.losses import *
from einops import rearrange
from uni3r.utils.camera_utils import get_scaled_camera
from uni3r.utils.cuda_splatting import DummyPipeline, render
from uni3r.utils.gaussian_model import GaussianModel
from torchmetrics import Accuracy, JaccardIndex, Metric
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from uni3r.utils.umeyama import umeyama, single_direction_chamfer_loss

from lpips import LPIPS
import time


import torch
from torchmetrics import Metric

class DepthEstimationMetric(Metric):
    full_state_update: bool = False

    def __init__(self, depth_cap=10.0, align_by_median=True, align_by_least_squares=False):
        super().__init__()
        self.add_state("abs_rel_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rmse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta2_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta3_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.depth_cap = depth_cap
        self.align_by_median = align_by_median
        self.align_by_least_squares = align_by_least_squares
        self.trim = 0.2  # trimming ratio for least squares

    @staticmethod
    def compute_scale_and_shift(pred, target, mask):
        a_00 = torch.sum(mask * pred * pred, dim=(1, 2))
        a_01 = torch.sum(mask * pred, dim=(1, 2))
        a_11 = torch.sum(mask, dim=(1, 2))

        b_0 = torch.sum(mask * pred * target, dim=(1, 2))
        b_1 = torch.sum(mask * target, dim=(1, 2))

        det = a_00 * a_11 - a_01 * a_01
        valid = det > 0

        scale = torch.zeros_like(b_0)
        shift = torch.zeros_like(b_1)

        scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return scale, shift

    def align_prediction(self, prediction, target, mask):
        # Ensure 3D: (B, H, W)
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(0)
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)

        B, H, W = prediction.shape
        aligned = torch.zeros_like(prediction)

        if self.align_by_median:
            for b in range(B):
                valid = mask[b] > 0
                scale = torch.median(target[b][valid] / prediction[b][valid])
                aligned[b] = prediction[b] * scale
        elif self.align_by_least_squares:
            for b in range(B):
                pred_b, target_b, mask_b = prediction[b], target[b], mask[b]
                scale, shift = self.compute_scale_and_shift(
                    pred_b.unsqueeze(0), target_b.unsqueeze(0), mask_b.unsqueeze(0)
                )
                pred_aligned = scale.view(1, 1) * pred_b + shift.view(1, 1)

                # trim and re-fit
                for _ in range(2):
                    err_map = torch.abs(pred_aligned - target_b) * mask_b
                    err_vals = err_map[mask_b.bool()]
                    if err_vals.numel() < 10:
                        break
                    sorted_err, _ = torch.sort(err_vals)
                    trim_thresh = sorted_err[int((1.0 - self.trim) * len(sorted_err))]
                    err_mask = (err_map < trim_thresh).float() * mask_b
                    scale, shift = self.compute_scale_and_shift(
                        pred_b.unsqueeze(0), target_b.unsqueeze(0), err_mask.unsqueeze(0)
                    )
                    pred_aligned = scale.view(1, 1) * pred_b + shift.view(1, 1)

                aligned[b] = torch.clamp(pred_aligned, max=self.depth_cap)
        else:
            aligned = prediction  # no alignment

        return aligned

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: predicted depth map, shape (N, H, W)
        targets: ground truth depth map, shape (N, H, W)
        """
        device = self.abs_rel_sum.device
        preds = preds.to(device)
        targets = targets.to(device)

        if preds.dim() == 2:
            preds = preds.unsqueeze(0)
            targets = targets.unsqueeze(0)

        mask = (targets > 0).to(preds.dtype)

        preds = preds * mask  # zero out invalid
        targets = targets * mask

        if mask.sum() == 0:
            return

        preds_aligned = self.align_prediction(preds, targets, mask)

        valid_mask = mask.bool()
        preds_valid = preds_aligned[valid_mask]
        targets_valid = targets[valid_mask]

        abs_rel = torch.mean(torch.abs(preds_valid - targets_valid) / targets_valid)
        rmse = torch.sqrt(torch.mean((preds_valid - targets_valid) ** 2))

        ratio = torch.max(preds_valid / targets_valid, targets_valid / preds_valid)
        delta1 = torch.mean((ratio < 1.03).float())
        delta2 = torch.mean((ratio < 1.03**2).float())
        delta3 = torch.mean((ratio < 1.03**3).float())

        n = preds_valid.numel()

        self.abs_rel_sum += abs_rel * n
        self.rmse_sum += rmse * n
        self.delta1_sum += delta1 * n
        self.delta2_sum += delta2 * n
        self.delta3_sum += delta3 * n
        self.total += n

    def compute(self):
        return {
            "AbsRel": self.abs_rel_sum / self.total,
            "RMSE": self.rmse_sum / self.total,
            "Delta1": self.delta1_sum / self.total,
            "Delta2": self.delta2_sum / self.total,
            "Delta3": self.delta3_sum / self.total,
        }


class KWRegr3D(Regr3D):
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None, **kwargs):
        return super().get_all_pts3d(gt1, gt2, pred1, pred2, dist_clip)

class L2Loss(LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance

class L1Loss(LLoss):
    """ Manhattan distance between 3d points """

    def distance(self, a, b):
        return torch.abs(a - b).mean()  # L1 distance


L2 = L2Loss()
L1 = L1Loss()


def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def merge_and_split_predictions(*args):
    merged = {}
    for key in [
        'scales', 'rotations', 'covs', 'opacities', 'sh_coeffs', 'means', 'gs_feats'
    ]:
        merged_pred = torch.stack([pred[key] for pred in args], dim=1)
        merged_pred = rearrange(merged_pred, 'b v h w ... -> b (v h w) ...')
        merged[key] = merged_pred

    # Split along the batch dimension
    batch_size = next(iter(merged.values())).shape[0]
    split = [{key: value[i] for key, value in merged.items()} for i in range(batch_size)]

    return split


class GaussianLoss(MultiLoss):

    def __init__(self,
                 ssim_weight=0.2,
                 feature_loss_weight=0.2,
                 geo_loss_weight=0.005,
                 labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.feature_loss_weight = feature_loss_weight
        self.geo_loss_weight = geo_loss_weight
        self.labels = labels
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        self.miou = JaccardIndex(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.accuracy = Accuracy(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.pipeline = DummyPipeline()
        # bg_color
        self.register_buffer('bg_color', torch.tensor([0.0, 0.0, 0.0]).cuda())

    def get_name(self):
        return f'GaussianLoss(ssim_weight={self.ssim_weight})'

    def compute_loss(self, gt, preds, target_view=None, model=None):
        pred = merge_and_split_predictions(*preds)
        if len(target_view) == 3:
            pred1 = preds[0]
            pred2 = preds[1]
        for i in range(len(pred)):
            pred[i]['extr'] = torch.stack([p['extr'][i] for p in preds])
            pred[i]['intr'] = torch.stack([p['intr'][i] for p in preds])

        # 3. render images(need gaussian model, camera, pipeline)
        rendered_images = []
        rendered_feats = []
        rendered_depths = []
        context1_depths = []
        context2_depths = []
        gt_images = []

        for i in range(len(pred)):
            # get gaussian model
            gaussians = GaussianModel.from_predictions(pred[i], sh_degree=3)
            # get camera
            target_view_list = target_view
            for j in range(len(target_view_list)):
                target_intrinsics = target_view_list[j]['camera_intrinsics'][i]
                target_extrinsics = target_view_list[j]['extrinsics'][i]  # actually is camera pose
                target_extrinsics_ = target_extrinsics.clone()
                target_extrinsics_[:3, :3] = target_extrinsics[:3, :3].mT
                target_extrinsics_[:3, 3:4] = -target_extrinsics_[:3, :3] @ target_extrinsics[:3, 3:4]  # actual extrinsics

                image_shape = target_view_list[j]['true_shape'][i]
                scale = 1  # scaling[i]
                camera = get_scaled_camera(None,
                                           target_extrinsics.detach(),
                                           target_intrinsics, scale,
                                           image_shape)
                # render(image and features)
                rendered_output = render(
                    camera,
                    gaussians,
                    self.pipeline,
                    self.bg_color,
                    intrinsics=target_intrinsics,
                    extrinsics=target_extrinsics_,
                    scale=target_view_list[j]['scale'][i]
                    )
                rendered_images.append(rendered_output['render'])
                rendered_feats.append(rendered_output['feature_map'])
                rendered_depths.append(rendered_output['depth'])
                if len(target_view_list) == 3:
                    if j == 0:
                        context1_depths.append(rendered_output['depth'] * 0.1)
                    if j == 1:
                        context2_depths.append(rendered_output['depth'] * 0.1)
                gt_images.append(target_view_list[j]['img'][i] * 0.5 + 0.5)

        rendered_images = torch.stack(rendered_images, dim=0)
        rendered_images = rendered_images.squeeze(1).permute(0, 3, 1, 2)  # B, 3, H, W
        gt_images = torch.stack(gt_images, dim=0)
        rendered_feats = torch.stack(rendered_feats, dim=0)
        rendered_feats = rendered_feats.squeeze(1).permute(0, 3, 1, 2)  # B, d_feats, H, W
        rendered_feats = model.feature_expansion(rendered_feats)  # B, 512, H//2, W//2
        rendered_depths = torch.stack(rendered_depths, dim=0).squeeze(1).permute(0, 3, 1, 2)
        gt_feats = model.lseg_feature_extractor.extract_features(gt_images)  # B, 512, H//2, W//2

        geo_loss = 0.0
        if len(target_view_list) == 3:
            # pointmap distiller loss here
            distiller_world_points = torch.concat((pred1['distiller_world_points'].unsqueeze(1), pred2['distiller_world_points'].unsqueeze(1)), dim=1)
            distiller_world_points_conf = torch.concat((pred1['distiller_world_points_conf'].unsqueeze(1), pred2['distiller_world_points_conf'].unsqueeze(1)), dim=1)
            gaussian_means = torch.concat((pred1['means'].unsqueeze(1), pred2['means'].unsqueeze(1)), dim=1)        
            geo_means = torch.nn.functional.interpolate(
                rearrange(distiller_world_points, "b v h w c -> (b v) c h w"),
                size=(256, 256),
                mode='nearest'             # 'nearest', 'bilinear', 'bicubic'
                # align_corners=False
            )
            geo_means = geo_means.view(distiller_world_points.shape[0], distiller_world_points.shape[1], 3, 256, 256)
            __, __, geo_loss = umeyama(geo_means, gaussian_means, Conf_map=distiller_world_points_conf, ratio=0.9)

        image_loss = torch.abs(rendered_images - gt_images).mean()
        image_loss += self.lpips_vgg(rendered_images, gt_images).mean() * 0.05
        feature_loss = (1 - torch.nn.functional.cosine_similarity(
            rendered_feats, gt_feats, dim=1)).mean()

        loss = image_loss + self.feature_loss_weight * feature_loss + self.geo_loss_weight * geo_loss

         
        return loss, {'image_loss': float(image_loss), 'feature_loss': float(feature_loss), 'geo_loss': float(geo_loss)}


class TestLoss(MultiLoss):

    def __init__(self,
                 pose_align_steps=False,
                 num_views=3,
                 labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
        super().__init__()
        self.labels = labels
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        self.lpips_scores = []
        self.miou = MeanIoU(
            num_classes=9,
            include_background=False,
            per_class=True,
            input_format= "index")
        # self.miou = MulticlassJaccardIndex(num_classes=9, ignore_index=0, average='none')
        # self.miou = JaccardIndex(
        #     num_classes=len(self.labels) + 1,
        #     task='multiclass',
        #     ignore_index=0)
        self.accuracy = Accuracy(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.pipeline = DummyPipeline()
        # bg_color
        self.register_buffer('bg_color', torch.tensor([0.0, 0.0, 0.0]).cuda())

        self.pose_align_steps = pose_align_steps
        if self.pose_align_steps:
            self.pose_embeds = nn.Embedding(num_views - 1, 9)
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
            torch.nn.init.zeros_(self.pose_embeds.weight)

        self.depth_metric = DepthEstimationMetric(align_by_median=False, align_by_least_squares=True)

    def get_name(self):
        return f'TestLoss'

    def update_lpips(self, pred, gt):
        score = self.lpips_vgg(pred.unsqueeze(0), gt.unsqueeze(0))
        self.lpips_scores.append(score.item())

    def compute_lpips_mean(self):
        return sum(self.lpips_scores) / len(self.lpips_scores)

    def compute_loss(self, gt, preds, target_view=None, model=None, pose_deltas=None, evaluate=True):
        pred = merge_and_split_predictions(*preds)
        
        # 添加与训练时相同的预测处理逻辑
        for i in range(len(pred)):
            pred[i]['extr'] = torch.stack([p['extr'][i] for p in preds])
            pred[i]['intr'] = torch.stack([p['intr'][i] for p in preds])

        rendered_images = []
        rendered_feats = []
        gt_images = []
        rendered_depths = []
        gt_depths = []


        for i in range(len(pred)):
            # get gaussian model
            gaussians = GaussianModel.from_predictions(pred[i], sh_degree=3)
            # get camera
            target_view_list = [target_view[-1]]  # 使用与训练时相同的数据选择策略
            # target_view_list = target_view
            for j in range(len(target_view_list)):
                # target_extrinsics = target_view_list[j]['camera_pose'][i]
                target_intrinsics = target_view_list[j]['camera_intrinsics'][i]
                target_extrinsics = target_view_list[j]['extrinsics'][i]  # actually camera pose
                target_extrinsics_ = target_extrinsics.clone()
                target_extrinsics_[:3, :3] = target_extrinsics[:3, :3].mT
                target_extrinsics_[:3, 3:4] = -target_extrinsics_[:3, :3] @ target_extrinsics[:3, 3:4]

                if pose_deltas is not None:
                    assert i == 0
                    pose_deltas_ = pose_deltas.weight[j].unsqueeze(0)
                    dx, drot = pose_deltas_[..., :3], pose_deltas_[..., 3:]
                    rot = rotation_6d_to_matrix(
                        drot + self.identity.expand(pose_deltas_.size(0), -1)
                    )  # (..., 3, 3)
                    transform = torch.eye(4, device=pose_deltas_.device).repeat((pose_deltas_.size(0), 1, 1))
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = dx
                    target_extrinsics_ = target_extrinsics_ @ transform.squeeze(0)

                image_shape = target_view_list[j]['true_shape'][i]
                scale = 1  # scaling[i]
                camera = get_scaled_camera(None,
                                           target_extrinsics.detach(),
                                           target_intrinsics, scale,
                                           image_shape)
                # render(image and features)
                rendered_output = render(
                    camera, 
                    gaussians, 
                    self.pipeline, 
                    self.bg_color,
                    intrinsics=target_intrinsics,
                    extrinsics=target_extrinsics_,
                    scale=target_view_list[j]['scale'][i]
                    )
                rendered_images.append(rendered_output['render'])
                rendered_feats.append(rendered_output['feature_map'])
                gt_images.append(target_view_list[j]['img'][i] * 0.5 + 0.5)
                rendered_depths.append(rendered_output['depth'])
                gt_depths.append(target_view_list[j]['depthmap'][i])

        rendered_images = torch.stack(rendered_images, dim=0)  # B, 3, H, W
        rendered_images = rendered_images.squeeze(1).permute(0, 3, 1, 2)
        gt_images = torch.stack(gt_images, dim=0)
        rendered_feats = torch.stack(rendered_feats, dim=0)  # B, d_feats, H, W
        rendered_feats = rendered_feats.squeeze(1).permute(0, 3, 1, 2)
        rendered_feats = model.feature_expansion(rendered_feats)  # B, 512, H//2, W//2
        gt_feats = model.lseg_feature_extractor.extract_features(gt_images)  # B, 512, H//2, W//2
        
        rendered_depths = torch.stack(rendered_depths, dim=0).squeeze(1).squeeze(-1)
        gt_depths = torch.stack(gt_depths, dim=0)

        image_loss = torch.abs(rendered_images - gt_images).mean()
        feature_loss = (1 - torch.nn.functional.cosine_similarity(
            rendered_feats, gt_feats, dim=1)).mean()

        logits = model.lseg_feature_extractor.decode_feature(rendered_feats, self.labels)
        pred = logits.argmax(dim=1, keepdim=True)
        pred = pred.clamp(max=7) + 1

        if evaluate:

            # # for context view
            # for i in range(len(target_view_list)-2):
            #     pred_cur = torch.where(target_view[i]["labelmap"] != 0, pred[i], 0)
            #     self.miou.update(pred_cur, target_view[i]["labelmap"].long())
            #     self.accuracy.update(pred_cur, target_view[i]["labelmap"].long())
            #     self.psnr.update(rendered_images[i], gt_images[i])
            #     self.ssim.update(rendered_images[i].unsqueeze(0), gt_images[i].unsqueeze(0))
            #     self.update_lpips(rendered_images[i], gt_images[i])
            #     rendered = rendered_depths[i]
            #     gt = gt_depths[i]
            #     device = rendered.device 
            #     self.depth_metric = self.depth_metric.to(rendered.device) 
            #     self.depth_metric.update(rendered.to(device), gt.to(device))
           
            # for target view
            for i in range(len(target_view_list)):
                pred_cur = torch.where(target_view[-i-1]["labelmap"] != 0, pred[-i-1], 0)
                self.miou.update(pred_cur, target_view[-i-1]["labelmap"].long())
                self.accuracy.update(pred_cur, target_view[-i-1]["labelmap"].long())
                self.psnr.update(rendered_images[-i-1], gt_images[-i-1])
                self.ssim.update(rendered_images[-i-1].unsqueeze(0), gt_images[-i-1].unsqueeze(0))
                self.update_lpips(rendered_images[-i-1], gt_images[-i-1])
                rendered = rendered_depths[-i-1]
                gt = gt_depths[-i-1]
                device = rendered.device 
                self.depth_metric = self.depth_metric.to(rendered.device) 
                self.depth_metric.update(rendered.to(device), gt.to(device))

        loss = image_loss  # + feature_loss
        return loss, {'image_loss': float(image_loss), 'feature_loss': float(feature_loss)}


# loss for one batch
def loss_of_one_batch(batch,
                      model,
                      criterion,
                      device,
                      symmetrize_batch=False,
                      use_amp=False,
                      ret=None,
                      total_time=None):

    context_views = []
    target_views = []
    total_time = []

    assert len(batch) != 0

    if len(batch) < 5:
        for i in range(len(batch)):
            if i < 2:
                context_views.append(batch[i])
            else:
                target_views.append(batch[i])
    else:
        for i in range(len(batch)):
            # if i < len(batch)-16: # using on test
            split_len = (len(batch) + 1) / 2
            if i < split_len:
                context_views.append(batch[i])
            else:
                target_views.append(batch[i])
    ignore_keys = set([
        'depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'
    ])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    # Get the actual model if it's distributed
    actual_model = model.module if hasattr(model, 'module') else model

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        # torch.cuda.synchronize()
        # time_start = time.perf_counter()
        if actual_model.training:
            pred = actual_model(context_views)
        else:
            with torch.no_grad():
                pred = actual_model(context_views)
        # torch.cuda.synchronize()
        # time_end = time.perf_counter()
        # elapsed = time_end - time_start
        # total_time.append(elapsed)

        if len(batch) < 5:
            render_views = context_views + target_views
        else:
            render_views = target_views

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            pose_align_steps = getattr(criterion, 'pose_align_steps', 0)
            if pose_align_steps:
                optimzer = torch.optim.Adam(criterion.pose_embeds.parameters(), lr=5e-3)
                criterion.pose_embeds.weight.data.zero_()
                for i in range(pose_align_steps):
                    if i != pose_align_steps - 1:
                        loss, _ = criterion(
                            context_views, pred, target_view=render_views, model=actual_model,
                            pose_deltas=criterion.pose_embeds, evaluate=False)
                        optimzer.zero_grad()
                        loss.backward()
                        optimzer.step()
                    else:
                        loss = criterion(
                            context_views, pred, target_view=render_views, model=actual_model,
                            pose_deltas=criterion.pose_embeds)
            else:
                loss = criterion(
                    context_views, pred, target_view=render_views, model=actual_model
                ) if criterion is not None else None

    result = dict(
        view=context_views,
        target_view=target_views,
        pred=pred,
        loss=loss)
    return result[ret] if ret else result

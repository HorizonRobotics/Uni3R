import numpy as np
import torch
from einops import rearrange
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance

def umeyama(X_all, Y_all, Conf_map=None, ratio=1.0):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """

    B, V, C, H, W = X_all.shape
    N = H*W

    if Conf_map is not None:
        Conf_map = torch.nn.functional.interpolate(Conf_map, X_all.shape[-2:], mode='nearest')
        Conf_map = rearrange(Conf_map, "b v h w -> b v (h w)")
        Conf_map = Conf_map.view(B*V, N)

    X_all = rearrange(X_all, "b v xyz h w -> b v (h w) xyz")
    Y_all = Y_all.view(B, V, N, C)
    X_all = X_all.view(B*V, N, C)
    Y_all = Y_all.view(B*V, N, C)

    if Conf_map is not None:
        K = int(ratio * Conf_map[0].numel())
        X_PM_all = torch.zeros((B*V, K, C), device=Y_all.device)
        Y_PM_all = torch.zeros((B*V, K, C), device=Y_all.device)
    else:
        X_PM_all = torch.zeros((B*V, N, C), device=Y_all.device)
        Y_PM_all = torch.zeros((B*V, N, C), device=Y_all.device)

    pm_loss = 0.0

    for i in range(B*V):

        X = X_all[i].permute(1,0).detach().cpu().numpy()
        Y = Y_all[i].permute(1,0).detach().cpu().numpy()

        if Conf_map is not None:
            threshold = torch.kthvalue(Conf_map[i], Conf_map[i].numel() - K + 1).values  # 70% as threshold
            mask = Conf_map[i] >= threshold
            mask = mask.detach().cpu()
            # X = X[:, mask]
            # Y = Y[:, mask]
        

        mu_x = X.mean(axis=1).reshape(-1, 1)
        mu_y = Y.mean(axis=1).reshape(-1, 1)
        var_x = np.square(X - mu_x).sum(axis=0).mean()
        cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
        U, D, VH = np.linalg.svd(cov_xy)
        S = np.eye(X.shape[0])
        if np.linalg.det(U) * np.linalg.det(VH) < 0:
            S[-1, -1] = -1
        c = np.trace(np.diag(D) @ S) / var_x
        R = U @ S @ VH
        t = mu_y - c * R @ mu_x

        X_PM = c * (R @ X) + t
        X_PM = torch.from_numpy(X_PM).permute(1,0).unsqueeze(0).to(torch.float32).to(Y_all.device)
        Y_PM = Y_all[i].unsqueeze(0)
        # Y_PM = torch.from_numpy(Y).permute(1,0).unsqueeze(0).to(torch.float32).to(Y_all.device)

        # import pdb; pdb.set_trace()

        # pm_loss += single_direction_chamfer_loss(Y_PM, X_PM)
        pm_loss += single_direction_chamfer_loss(Y_PM[:, mask, :], X_PM[:, mask, :])

    if pm_loss != 0.0:
        pm_loss = pm_loss / (B*V)

    # print('pm_loss:', pm_loss)

        # X_PM_all[i] = X_PM
        # Y_PM_all[i] = Y_PM

    # pm_loss = single_direction_chamfer_loss(Y_PM_all, X_PM_all)

    return X_PM_all, Y_all, pm_loss

def single_direction_chamfer_loss(pred_points, target_points):
    # loss_bidirectional, _ = chamfer_distance(pred_points, target_points)
    loss_sindirectional, _ = chamfer_distance(pred_points, target_points, single_directional=True)
    return loss_sindirectional
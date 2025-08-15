import torch
import matplotlib.pyplot as plt


def visualize(x, filename):
    if isinstance(x, list):
        x = torch.stack(x)  # B, 3/1, H, W
    if x.dim() == 3:
        x = x.unsqueeze(1)
    assert x.size(1) in (1, 3)
    x = x.permute(2, 0, 3, 1).flatten(1, 2).detach().cpu().numpy()
    if x.shape[-1] == 1:
        from matplotlib import cm
        cmap = cm.get_cmap('Spectral_r')
        x -= x.min()
        x /= x.max()
        x = cmap(x)[..., 0, :3]
    x -= x.min()
    x /= x.max()
    plt.imsave(filename + '.png', x)

import torch
import torch.nn as nn
from submodules.dust3r.dust3r.model import AsymmetricCroCo3DStereo

class Dust3RWithFeature(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(**kwargs)
        self.dust3r.set_freeze(kwargs['freeze'])

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.dust3r._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self.dust3r._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self.dust3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self.dust3r._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        
        enc_feat = torch.cat([feat1, feat2], dim=1) # B,2S,D
        return (res1, res2), enc_feat
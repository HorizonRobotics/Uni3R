from submodules.PointTransformerV3.model import *
from submodules.dust3r.croco.models.blocks import Mlp

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, point, feature):
        point_feat = point.feat
        
        # Calculate number of points for each sample
        batch_size = len(point.offset)
        sample_lengths = torch.diff(point.offset, prepend=torch.tensor([0], device=point.offset.device))
        max_length = sample_lengths.max()
        
        # Create padded feature tensor and attention mask
        padded_feat = torch.zeros((batch_size, max_length, point_feat.shape[1]), 
                                device=point_feat.device)
        attention_mask = torch.zeros((batch_size, max_length), 
                                   device=point_feat.device, dtype=torch.bool)
        
        # Pad features and create mask
        start_idx = 0
        for i in range(batch_size):
            curr_length = sample_lengths[i]
            padded_feat[i, :curr_length] = point_feat[start_idx:start_idx + curr_length]
            attention_mask[i, :curr_length] = True
            start_idx += curr_length
        
        # Self-attention
        normed_feat = self.norm1(padded_feat)
        padded_feat = padded_feat + self.drop_path(
            self.attn(
                query=normed_feat,
                key=normed_feat,
                value=normed_feat,
                key_padding_mask=~attention_mask
            )[0]
        )
        
        # Cross-attention with external feature
        feature = self.norm_y(feature)  # [B, L, C]
        padded_feat = padded_feat + self.drop_path(
            self.cross_attn(
                query=self.norm2(padded_feat),
                key=feature,
                value=feature
            )[0]
        )
        
        # MLP block
        padded_feat = padded_feat + self.drop_path(
            self.mlp(self.norm3(padded_feat))
        )
        
        # Restore to original format
        point_feat = torch.cat([
            padded_feat[i, :sample_lengths[i]] for i in range(batch_size)
        ])
        
        point.feat = point_feat
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point
        
class PTV3(PointTransformerV3):
    def __init__(self, 
                 cross_dust=False, 
                 cross_lseg=False, 
                 cross_multi_scale=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.cross_dust = cross_dust
        self.cross_lseg = cross_lseg
        self.cross_multi_scale = cross_multi_scale

        if cross_dust:
            self.dust_feat_proj = nn.Linear(1024, 512)
            self.decoder_block_dust = DecoderBlock(dim=512, num_heads=8)
        if cross_lseg:
            self.lseg_feat_proj = nn.Linear(512, 512)
            self.decoder_block_lseg = DecoderBlock(dim=512, num_heads=8)
        if cross_multi_scale:
            self.multi_scale_proj = nn.Linear(512, 256)
            self.decoder_block_multi_scale = DecoderBlock(dim=256, num_heads=4)

    def forward(self, data_dict, dust3r_feature, lseg_feature, multi_scale_feature):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if self.cross_dust:
            dust3r_feature = self.dust_feat_proj(dust3r_feature)
            point = self.decoder_block_dust(point, dust3r_feature)
        if self.cross_lseg:
            lseg_feature = self.lseg_feat_proj(lseg_feature)
            point = self.decoder_block_lseg(point, lseg_feature)
        if not self.cls_mode:
            for i, dec_block in enumerate(self.dec):
                point = dec_block(point)
                if i == 0 and self.cross_multi_scale:
                    multi_scale_feature = self.multi_scale_proj(multi_scale_feature)
                    point = self.decoder_block_multi_scale(point, multi_scale_feature)
        return point
    
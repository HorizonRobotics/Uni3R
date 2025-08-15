from dataclasses import dataclass

@dataclass
class DUSt3RConfig:
    pretrained_model_name_or_path: str
    # ... other dust3r specific configs

@dataclass
class PointTransformerConfig:
    enc_depths: list[int]
    enc_channels: list[int]
    enc_num_head: list[int]
    enc_patch_size: list[int]
    dec_depths: list[int]
    dec_channels: list[int]
    dec_num_head: list[int]
    dec_patch_size: list[int]
    # ... other point transformer specific configs

@dataclass
class GaussianHeadConfig:
    rgb_residual: bool
    d_gs_feats: int = 64
    # ... other gaussian head specific configs

@dataclass
class LSegConfig:
    pretrained_model_name_or_path: str
    half_res: bool
    device: str = 'cuda'
    # ... other lseg specific configs

@dataclass
class LSMConfig:
    dust3r_config: DUSt3RConfig
    point_transformer_config: PointTransformerConfig
    gaussian_head_config: GaussianHeadConfig
    lseg_config: LSegConfig
    freeze_dust3r: bool = True
    freeze_lseg: bool = True
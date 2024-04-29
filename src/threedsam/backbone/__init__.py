from .resnet_fpn import ResNetFPN_32

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        return ResNetFPN_32(config['resnetfpn'])
    else:
        raise ValueError(f"Not supported backbone type.")
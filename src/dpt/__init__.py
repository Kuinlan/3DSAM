import torch

from .models import DPTDepthModel


def init_dpt(dpt_weight_path,
             scale=0.000305,
             shift=0.1378,
             invert=True,
             backbone="vitb_rn50_384",
             non_negative=True,
             enable_attention_hooks=False,
             frozen=False):

    dpt_model = DPTDepthModel(
        path=dpt_weight_path,
        scale=scale,
        shift=shift,
        invert=invert,
        backbone=backbone,
        non_negative=non_negative,
        enable_attention_hooks=enable_attention_hooks
    )

    dpt_model.eval()
    dpt_model = dpt_model.to(memory_format=torch.channels_last)
    dpt_model = dpt_model.half()
    
    if frozen:
        for params in dpt_model.parameters():
            params.requires_grad = False
    
    return dpt_model
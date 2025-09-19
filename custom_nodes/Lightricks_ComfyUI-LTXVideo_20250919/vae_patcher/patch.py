import types

import torch

from .forwards import (
    block_forward,
    decoder_forward,
    res_block_forward,
    upsample_forward,
)


def patch_vae(vae_model, patch_block=4):

    vae_model.decoder.forward = types.MethodType(decoder_forward, vae_model.decoder)
    vae_model.decoder.conv_out.conv.weight.data = (
        vae_model.decoder.conv_out.conv.weight.data.to(
            memory_format=torch.channels_last_3d
        )
    )

    for i, block in enumerate(reversed(vae_model.decoder.up_blocks)):
        if i >= patch_block:
            break
        if block.__class__.__name__ == "UNetMidBlock3D":
            block.forward = types.MethodType(block_forward, block)
            for name, param in block.named_parameters():
                if "conv1" in name or "conv2" in name:
                    if "weight" in name:
                        param = param.to(memory_format=torch.channels_last_3d)
            for res_block in block.res_blocks:
                if res_block.__class__.__name__ == "ResnetBlock3D":
                    res_block.forward = types.MethodType(res_block_forward, res_block)
        elif block.__class__.__name__ == "DepthToSpaceUpsample":
            block.forward = types.MethodType(upsample_forward, block)

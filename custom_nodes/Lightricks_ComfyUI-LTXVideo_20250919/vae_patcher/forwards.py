from typing import Optional

import numpy as np
import torch
from einops import rearrange

from .kernels.ops import add_inplace, pixel_norm_inplace


def find_split_size(F: int) -> int:
    s = 16
    while F % s < 3:
        s -= 1
    return s


def inplace_patch_conv_2(x, conv: torch.nn.Conv3d):

    assert conv.kernel_size == (3, 3, 3)
    assert conv.stride == (1, 1, 1)
    assert conv.padding == (0, 1, 1)
    assert conv.dilation == (1, 1, 1)
    assert conv.groups == 1

    x[:, :, 0, :, :] = x[:, :, 1, :, :].clone()
    x[:, :, -1, :, :] = x[:, :, -2, :, :].clone()

    if x.shape[2] > 16:
        split_size = find_split_size(x.shape[2])
        num_splits = (x.shape[2] + split_size - 1) // split_size
    else:
        split_size = x.shape[2] - 1
        num_splits = 1

    out_channels = conv.out_channels
    in_channels = conv.in_channels
    x_buffers = torch.empty(
        x.shape[0],
        x.shape[1],
        1,
        x.shape[3],
        x.shape[4],
        device=x.device,
        dtype=x.dtype,
    )
    o_buffers = torch.empty(
        x.shape[0],
        x.shape[1],
        1,
        x.shape[3],
        x.shape[4],
        device=x.device,
        dtype=x.dtype,
    )
    # 0 case
    x_buffers[:, :, 0, :, :] = x[:, :, split_size - 1, :, :].clone()
    x[:, :out_channels, 1:split_size, :, :] = conv(
        x[:, :in_channels, : split_size + 1, :, :]
    )

    for i in range(1, num_splits):
        curr_frame_start = i * split_size
        curr_frame_end = min((i + 1) * split_size, x.shape[2] - 1)
        o_buffers[:, :, 0, :, :] = x[:, :, curr_frame_start - 1, :, :].clone()
        x[:, :, curr_frame_start - 1, :, :] = x_buffers[:, :, 0, :, :].clone()
        x_buffers[:, :, 0, :, :] = x[:, :, curr_frame_end - 1, :, :].clone()
        x[:, :out_channels, curr_frame_start:curr_frame_end, :, :] = conv(
            x[:, :in_channels, curr_frame_start - 1 : curr_frame_end + 1, :, :]
        )
        x[:, :, curr_frame_start - 1, :, :] = o_buffers[:, :, 0, :, :].clone()


def res_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    causal: bool = True,
    timestep: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:

    batch_size = hidden_states.shape[0]
    ada_values = self.scale_shift_table[None, ..., None, None, None].to(
        timestep.device
    ) + timestep.reshape(
        batch_size,
        4,
        -1,
        timestep.shape[-3],
        timestep.shape[-2],
        timestep.shape[-1],
    )
    shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)
    pixel_norm_inplace(hidden_states, scale1, shift1, 1e-9)
    inplace_patch_conv_2(hidden_states, self.conv1.conv)
    pixel_norm_inplace(hidden_states, scale2, shift2, 1e-9)
    inplace_patch_conv_2(hidden_states, self.conv2.conv)


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    causal: bool = True,
    timestep: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    timestep_embed = None
    if self.timestep_conditioning:
        assert (
            timestep is not None
        ), "should pass timestep with timestep_conditioning=True"
        batch_size = hidden_states.shape[0]
        timestep_embed = self.time_embedder(
            timestep=timestep.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        timestep_embed = timestep_embed.view(
            batch_size, timestep_embed.shape[-1], 1, 1, 1
        )
    workspace = torch.empty(
        hidden_states.shape[0],
        hidden_states.shape[1],
        hidden_states.shape[2] + 2,
        hidden_states.shape[3],
        hidden_states.shape[4],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        memory_format=torch.channels_last_3d,
    )
    for resnet in self.res_blocks:
        workspace[:, :, 1:-1].copy_(hidden_states)
        resnet(workspace, causal=causal, timestep=timestep_embed)
        add_inplace(hidden_states, workspace, 1)
    del workspace
    torch.cuda.empty_cache()
    return hidden_states


def unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


def upsample_forward(self, x, causal: bool = True):
    if self.residual:
        # Reshape and duplicate the input to match the output shape
        x_in = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        num_repeat = np.prod(self.stride) // self.out_channels_reduction_factor
        x_in = x_in.repeat(1, num_repeat, 1, 1, 1)

    workspace = torch.empty(
        x.shape[0],
        self.conv.conv.out_channels,
        x.shape[2] + 2,
        x.shape[3],
        x.shape[4],
        device=x.device,
        dtype=x.dtype,
    )
    workspace[:, : x.shape[1], 1:-1, :, :].copy_(x)
    del x
    torch.cuda.empty_cache()
    inplace_patch_conv_2(workspace, self.conv.conv)
    x = workspace[:, : self.conv.conv.out_channels, 1:-1, :, :]

    x = rearrange(
        x,
        "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
        p1=self.stride[0],
        p2=self.stride[1],
        p3=self.stride[2],
    )
    if self.residual:
        x.add_(x_in)
    del x_in
    torch.cuda.empty_cache()
    return x[:, :, 1:, :, :]


def decoder_forward(
    self,
    sample: torch.FloatTensor,
    timestep: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    r"""The forward method of the `Decoder` class."""
    batch_size = sample.shape[0]

    sample = self.conv_in(sample, causal=self.causal)

    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = sample.to(upscale_dtype)

    if self.timestep_conditioning:
        assert (
            timestep is not None
        ), "should pass timestep with timestep_conditioning=True"
        scaled_timestep = (timestep * self.timestep_scale_multiplier).to(sample.device)

    for up_block in self.up_blocks:
        if (
            self.timestep_conditioning
            and up_block.__class__.__name__ == "UNetMidBlock3D"
        ):
            sample = up_block(sample, causal=self.causal, timestep=scaled_timestep)
        else:
            sample = up_block(sample, causal=self.causal)

    if self.timestep_conditioning:
        embedded_timestep = self.last_time_embedder(
            timestep=scaled_timestep.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=sample.shape[0],
            hidden_dtype=sample.dtype,
        )
        embedded_timestep = embedded_timestep.view(
            batch_size, embedded_timestep.shape[-1], 1, 1, 1
        )
        ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
            embedded_timestep.device
        ) + embedded_timestep.reshape(
            batch_size,
            2,
            -1,
            embedded_timestep.shape[-3],
            embedded_timestep.shape[-2],
            embedded_timestep.shape[-1],
        )
        shift, scale = ada_values.unbind(dim=1)

    workspace = torch.empty(
        sample.shape[0],
        sample.shape[1],
        sample.shape[2] + 2,
        sample.shape[3],
        sample.shape[4],
        device=sample.device,
        dtype=sample.dtype,
        memory_format=torch.channels_last_3d,
    )
    workspace[:, :, 1:-1, :, :].copy_(sample)
    del sample
    torch.cuda.empty_cache()

    pixel_norm_inplace(workspace, scale, shift, 1e-9)
    inplace_patch_conv_2(workspace, self.conv_out.conv)

    return unpatchify(
        workspace[:, : self.conv_out.conv.out_channels, 1:-1],
        patch_size_hw=self.patch_size,
        patch_size_t=1,
    )

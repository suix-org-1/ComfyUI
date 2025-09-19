from typing import Optional

import folder_paths
import torch
import torch.nn as nn
from comfy import model_management
from diffusers import ConfigMixin, ModelMixin
from einops import rearrange

from .nodes_registry import comfy_node


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        b, c, f, h, w = x.shape
        r = self.r
        out_c = c // (r**3)
        x = x.view(b, out_c, r, r, r, f, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)  # (b, out_c, f, r, h, r, w, r)
        x = x.reshape(b, out_c, f * r, h * r, w * r)
        return x


class PixelShuffle2D(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        r = self.r
        out_c = c // (r * r)
        x = x.view(b, out_c, r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)  # (b, out_c, h, r, w, r)
        x = x.reshape(b, out_c, h * r, w * r)
        return x


class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        b, c, f, h, w = x.shape
        r = self.r
        out_c = c // r
        x = x.view(b, out_c, r, f, h, w)  # [B, C//r, r, F, H, W]
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, C//r, F, r, H, W]
        x = x.reshape(b, out_c, f * r, h, w)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, channels: int, mid_channels: Optional[int] = None, dims: int = 3
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.conv1 = Conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, mid_channels)
        self.conv2 = Conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


class LatentUpsampler(ModelMixin, ConfigMixin):
    """
    Model to spatially upsample VAE latents.

    Args:
        in_channels (`int`): Number of channels in the input latent
        mid_channels (`int`): Number of channels in the middle layers
        num_blocks_per_stage (`int`): Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`): Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`): Whether to spatially upsample the latent
        temporal_upsample (`bool`): Whether to temporally upsample the latent
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample

        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.initial_conv = Conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffle3D(2),
            )
        elif spatial_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                PixelShuffle2D(2),
            )
        elif temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffle1D(2),
            )
        else:
            raise ValueError(
                "Either spatial_upsample or temporal_upsample must be True"
            )

        self.post_upsample_res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = Conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                x = x[:, :, 1:, :, :]
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels=config.get("in_channels", 4),
            mid_channels=config.get("mid_channels", 128),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            dims=config.get("dims", 2),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
        )

    def config(self):
        return {
            "_class_name": "LatentUpsampler",
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "num_blocks_per_stage": self.num_blocks_per_stage,
            "dims": self.dims,
            "spatial_upsample": self.spatial_upsample,
            "temporal_upsample": self.temporal_upsample,
        }

    def load_weights(self, weights_path: str) -> None:
        """
        Load model weights from a .safetensors file and switch to evaluation mode.

        Args:
            weights_path (str): Path to the .safetensors file containing the model weights

        Raises:
            RuntimeError: If there are missing or unexpected keys in the state dict
        """
        import safetensors.torch

        sd = safetensors.torch.load_file(weights_path)
        self.load_state_dict(sd, strict=False, assign=True)
        # Switch to evaluation mode
        self.eval()


@comfy_node(name="LTXVLatentUpsampler")
class LTXVLatentUpsampler:
    """
    Upsamples a video latent by a factor of 2.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_model": ("UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upsample_latent"
    CATEGORY = "latent/video"

    def upsample_latent(
        self, samples: dict, upscale_model: LatentUpsampler, vae
    ) -> tuple:
        """
        Upsample the input latent using the provided model.

        Args:
            samples (dict): Input latent samples
            upscale_model (LatentUpsampler): Loaded upscale model

        Returns:
            tuple: Tuple containing the upsampled latent
        """
        latents = samples["samples"]

        # Ensure latents are on the same device as the model
        if latents.device != upscale_model.device:
            latents = latents.to(upscale_model.device)
        latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
        upsampled_latents = upscale_model(latents)
        upsampled_latents = vae.first_stage_model.per_channel_statistics.normalize(
            upsampled_latents
        )
        upsampled_latents = upsampled_latents.to(model_management.intermediate_device())
        return_dict = samples.copy()
        return_dict["samples"] = upsampled_latents
        return (return_dict,)


@comfy_node(name="LTXVLatentUpsamplerModelLoader")
class LTXVLatentUpsamplerModelLoader:
    """
    Loads a latent upsampler model from a .safetensors file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "spatial_upsample": ("BOOLEAN", {"default": True}),
                "temporal_upsample": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "latent/video"

    def load_model(
        self, upscale_model: str, spatial_upsample: bool, temporal_upsample: bool
    ) -> tuple:
        """
        Load the upscale model from the specified file.

        Args:
            upscale_model (str): Name of the upscale model file

        Returns:
            tuple: Tuple containing the loaded model
        """
        upscale_model_path = folder_paths.get_full_path("upscale_models", upscale_model)
        if upscale_model_path is None:
            raise ValueError(f"Upscale model {upscale_model} not found")

        try:
            latent_upsampler = LatentUpsampler(
                num_blocks_per_stage=4,
                dims=3,
                spatial_upsample=spatial_upsample,
                temporal_upsample=temporal_upsample,
            )
            latent_upsampler.load_weights(upscale_model_path)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize LatentUpsampler with this configuration: {str(e)}"
            )

        latent_upsampler.eval()
        # Move model to appropriate device
        device = model_management.get_torch_device()
        latent_upsampler.to(device)

        return (latent_upsampler,)

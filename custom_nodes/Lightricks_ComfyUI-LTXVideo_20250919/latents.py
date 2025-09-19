from typing import Optional

import comfy_extras.nodes_lt as nodes_lt
import torch

from .nodes_registry import comfy_node


@comfy_node(name="LTXVSelectLatents")
class LTXVSelectLatents:
    """
    Selects a range of frames from a video latent.

    Features:
    - Supports positive and negative indexing
    - Preserves batch processing capabilities
    - Handles noise masks if present
    - Maintains 5D tensor format
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "start_index": (
                    "INT",
                    {"default": 0, "min": -9999, "max": 9999, "step": 1},
                ),
                "end_index": (
                    "INT",
                    {"default": -1, "min": -9999, "max": 9999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "select_latents"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Selects a range of frames from the video latent. "
        "start_index and end_index define a closed interval (inclusive of both endpoints)."
    )

    def select_latents(self, samples: dict, start_index: int, end_index: int) -> tuple:
        """
        Selects a range of frames from the video latent.

        Args:
            samples (dict): Video latent dictionary
            start_index (int): Starting frame index (supports negative indexing)
            end_index (int): Ending frame index (supports negative indexing)

        Returns:
            tuple: Contains modified latent dictionary with selected frames

        Raises:
            ValueError: If indices are invalid
        """
        try:
            s = samples.copy()
            video_latent = s["samples"]
            batch, channels, frames, height, width = video_latent.shape

            # Handle negative indices
            start_idx = frames + start_index if start_index < 0 else start_index
            end_idx = frames + end_index if end_index < 0 else end_index

            # Validate and clamp indices
            start_idx = max(0, min(start_idx, frames - 1))
            end_idx = max(0, min(end_idx, frames - 1))
            if start_idx > end_idx:
                start_idx = min(start_idx, end_idx)

            # Select frames while maintaining 5D format
            s["samples"] = video_latent[:, :, start_idx : end_idx + 1, :, :]

            # Handle noise mask if present
            if "noise_mask" in s:
                s["noise_mask"] = s["noise_mask"][:, :, start_idx : end_idx + 1, :, :]

            return (s,)

        except Exception as e:
            print(f"[LTXVSelectLatents] Error: {str(e)}")
            raise


@comfy_node(name="LTXVAddLatents")
class LTXVAddLatents:
    """
    Concatenates two video latents along the frames dimension.

    Features:
    - Validates dimension compatibility
    - Handles device placement
    - Preserves noise masks with proper handling
    - Supports batch processing
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents1": ("LATENT",),
                "latents2": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_latents"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Concatenates two video latents along the frames dimension. "
        "latents1 and latents2 must have the same dimensions except for the frames dimension."
    )

    def add_latents(
        self, latents1: torch.Tensor, latents2: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenates two video latents along the frames dimension.

        Args:
            latents1 (dict): First video latent dictionary
            latents2 (dict): Second video latent dictionary

        Returns:
            tuple: Contains concatenated latent dictionary

        Raises:
            ValueError: If latent dimensions don't match
            RuntimeError: If tensor operations fail
        """
        try:
            s = latents1.copy()
            video_latent1 = latents1["samples"]
            video_latent2 = latents2["samples"]

            # Ensure tensors are on the same device
            target_device = video_latent1.device
            video_latent2 = video_latent2.to(target_device)

            # Validate dimensions
            self._validate_dimensions(video_latent1, video_latent2)

            # Concatenate along frames dimension
            s["samples"] = torch.cat([video_latent1, video_latent2], dim=2)

            # Handle noise masks
            s["noise_mask"] = self._merge_noise_masks(
                latents1, latents2, video_latent1.shape[2], video_latent2.shape[2]
            )

            return (s,)

        except Exception as e:
            print(f"[LTXVAddLatents] Error: {str(e)}")
            raise

    def _validate_dimensions(self, latent1: torch.Tensor, latent2: torch.Tensor):
        """Validates that latent dimensions match except for frames."""
        b1, c1, f1, h1, w1 = latent1.shape
        b2, c2, f2, h2, w2 = latent2.shape

        if not (b1 == b2 and c1 == c2 and h1 == h2 and w1 == w2):
            raise ValueError(
                f"Latent dimensions must match (except frames dimension).\n"
                f"Got shapes {latent1.shape} and {latent2.shape}"
            )

    def _merge_noise_masks(
        self, latents1: torch.Tensor, latents2: torch.Tensor, frames1: int, frames2: int
    ) -> Optional[torch.Tensor]:
        """Merges noise masks from both latents with proper handling."""
        if "noise_mask" in latents1 and "noise_mask" in latents2:
            return torch.cat([latents1["noise_mask"], latents2["noise_mask"]], dim=2)
        elif "noise_mask" in latents1:
            zeros = torch.zeros_like(latents1["noise_mask"][:, :, :frames2, :, :])
            return torch.cat([latents1["noise_mask"], zeros], dim=2)
        elif "noise_mask" in latents2:
            zeros = torch.zeros_like(latents2["noise_mask"][:, :, :frames1, :, :])
            return torch.cat([zeros, latents2["noise_mask"]], dim=2)
        return None


@comfy_node(name="LTXVSetVideoLatentNoiseMasks")
class LTXVSetVideoLatentNoiseMasks:
    """
    Applies multiple masks to a video latent.

    Features:
    - Supports multiple input mask formats (2D, 3D, 4D)
    - Automatically handles fewer masks than frames by reusing the last mask
    - Resizes masks to match latent dimensions
    - Preserves batch processing capabilities

    Input Formats:
    - 2D mask: Single mask [H, W]
    - 3D mask: Multiple masks [M, H, W]
    - 4D mask: Multiple masks with channels [M, C, H, W]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Applies multiple masks to a video latent. "
        "masks can be 2D, 3D, or 4D tensors. "
        "If there are fewer masks than frames, the last mask will be reused."
    )

    def set_mask(self, samples: dict, masks: torch.Tensor) -> tuple:
        """
        Applies masks to video latent frames.

        Args:
            samples (dict): Video latent dictionary containing 'samples' tensor
            masks (torch.Tensor): Mask tensor in various possible formats
                - 2D: [H, W] single mask
                - 3D: [M, H, W] multiple masks
                - 4D: [M, C, H, W] multiple masks with channels

        Returns:
            tuple: Contains modified latent dictionary with applied masks

        Raises:
            ValueError: If mask dimensions are unsupported
            RuntimeError: If tensor operations fail
        """
        try:
            s = samples.copy()
            video_latent = s["samples"]
            batch_size, channels, num_frames, height, width = video_latent.shape

            # Initialize noise_mask if not present
            if "noise_mask" not in s:
                s["noise_mask"] = torch.zeros(
                    (batch_size, 1, num_frames, height, width),
                    dtype=video_latent.dtype,
                    device=video_latent.device,
                )

            # Process masks
            masks_reshaped = self._reshape_masks(masks)
            M = masks_reshaped.shape[0]
            resized_masks = self._resize_masks(masks_reshaped, height, width)

            # Apply masks efficiently
            self._apply_masks(s["noise_mask"], resized_masks, num_frames, M)
            return (s,)

        except Exception as e:
            print(f"[LTXVSetVideoLatentNoiseMasks] Error: {str(e)}")
            raise

    def _reshape_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Reshapes input masks to consistent 4D format."""
        original_shape = tuple(masks.shape)
        ndims = masks.ndim

        if ndims == 2:
            return masks.unsqueeze(0).unsqueeze(0)
        elif ndims == 3:
            return masks.reshape(masks.shape[0], 1, masks.shape[1], masks.shape[2])
        elif ndims == 4:
            return masks.reshape(masks.shape[0], 1, masks.shape[2], masks.shape[3])
        else:
            raise ValueError(
                f"Unsupported 'masks' dimension: {original_shape}. "
                "Must be 2D (H,W), 3D (M,H,W), or 4D (M,C,H,W)."
            )

    def _resize_masks(
        self, masks: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Resizes all masks to match latent dimensions."""
        return torch.nn.functional.interpolate(
            masks, size=(height, width), mode="bilinear", align_corners=False
        )

    def _apply_masks(
        self,
        noise_mask: torch.Tensor,
        resized_masks: torch.Tensor,
        num_frames: int,
        M: int,
    ) -> None:
        """Applies resized masks to all frames."""
        for f in range(num_frames):
            mask_idx = min(f, M - 1)  # Reuse last mask if we run out
            noise_mask[:, :, f] = resized_masks[mask_idx]


@comfy_node(name="LTXVAddLatentGuide")
class LTXVAddLatentGuide(nodes_lt.LTXVAddGuide):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "guiding_latent": ("LATENT",),
                "latent_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Latent index to start the conditioning at. Can be negative to"
                        "indicate that the conditioning is on the frames before the latent.",
                    },
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "ltxtricks"
    FUNCTION = "generate"

    DESCRIPTION = "Adds a keyframe or a video segment at a specific frame index."

    def generate(
        self, vae, positive, negative, latent, guiding_latent, latent_idx, strength
    ):
        noise_mask = nodes_lt.get_noise_mask(latent)
        latent = latent["samples"]
        guiding_latent = guiding_latent["samples"]
        scale_factors = vae.downscale_index_formula

        if latent_idx <= 0:
            frame_idx = latent_idx * scale_factors[0]
        else:
            frame_idx = 1 + (latent_idx - 1) * scale_factors[0]

        positive, negative, latent, noise_mask = self.append_keyframe(
            positive=positive,
            negative=negative,
            frame_idx=frame_idx,
            latent_image=latent,
            noise_mask=noise_mask,
            guiding_latent=guiding_latent,
            strength=strength,
            scale_factors=scale_factors,
        )

        return (
            positive,
            negative,
            {"samples": latent, "noise_mask": noise_mask},
        )

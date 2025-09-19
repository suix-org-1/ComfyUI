import comfy
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch

try:
    from q8_kernels.functional.ops import hadamard_transform
    from q8_kernels.integration.patch_transformer import (
        patch_comfyui_native_transformer,
        patch_comfyui_transformer,
    )

    Q8_AVAILABLE = True
except ImportError:
    Q8_AVAILABLE = False

from .nodes_registry import comfy_node


def check_q8_available():
    if not Q8_AVAILABLE:
        raise ImportError(
            "Q8 kernels are not available. To use this feature install the q8_kernels package from here:."
            "https://github.com/Lightricks/LTX-Video-Q8-Kernels"
        )


@comfy_node(name="LTXQ8Patch")
class LTXVQ8Patch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "use_fp8_attention": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use FP8 attention."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Q8 Patcher"

    def patch(self, model, use_fp8_attention):
        check_q8_available()
        m = model.clone()
        diffusion_key = "diffusion_model"
        diffusion_model = m.get_model_object(diffusion_key)
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            transformer_key = "diffusion_model.transformer"
            patcher = patch_comfyui_transformer
        else:
            transformer_key = "diffusion_model"
            patcher = patch_comfyui_native_transformer
        transformer = m.get_model_object(transformer_key)
        patcher(transformer, use_fp8_attention, True)
        m.add_object_patch(transformer_key, transformer)
        return (m,)


def idendity_quant_fn(x, t):
    return x.to(dtype=t)


@comfy_node(name="LTXVQ8LoraModelLoader")
class LTXVQ8LoraModelLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "lightricks/LTXV"
    FUNCTION = "load_lora_model_only"

    def load_lora(self, model, lora_name, strength_model):
        quant_fn = hadamard_transform
        is_patched_transformer = getattr(
            model.get_model_object("diffusion_model"), "is_patched", False
        )
        if not is_patched_transformer or not Q8_AVAILABLE:
            raise ValueError(
                "LTXV Q8 Patcher is not applied to the model. Please use LTXQ8Patch node before loading lora or install q8_kernels."
            )

        if strength_model == 0:
            return model

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            new_lora = {}
            for k in lora:
                device = lora[k].device
                if lora[k].ndim == 2:
                    if "lora_A" in k:
                        new_lora[k] = quant_fn(
                            lora[k].to(device="cuda", dtype=torch.bfloat16),
                            out_type=torch.bfloat16,
                        ).to(device)
                    else:
                        new_lora[k] = lora[k]
            self.loaded_lora = (lora_path, new_lora)

        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, new_lora, strength_model, 0
        )
        return model_lora

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, lora_name, strength_model),)

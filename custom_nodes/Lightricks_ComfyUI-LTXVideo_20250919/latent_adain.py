import copy

import torch

from .nodes_registry import comfy_node


@comfy_node(name="LTXVAdainLatent")
class LTXVAdainLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "reference": ("LATENT",),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "Lightricks/latents"

    def batch_normalize(self, latents, reference, factor):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"]  #  B x C x F x H x W

        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                r_sd, r_mean = torch.std_mean(
                    reference["samples"][i, c], dim=None
                )  # index by original dim order
                i_sd, i_mean = torch.std_mean(t[i, c], dim=None)

                t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean

        latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
        return (latents_copy,)

import torch
import numpy as np
from PIL import Image

class CompositorColorPicker:
    """
    This node converts RGB values (0-255) to various color formats:
    - Hex color string (#RRGGBB)
    - 16-bit color value (RGB565 or RGB555 format)
    - 24-bit color value (compatible with ComfyUI's emptyImage node)
    - Preview image of the color
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "red": ("INT", {"default": 0, "min": 0, "max": 255}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255}),
                "format": (["RGB565", "RGB555"], {"default": "RGB565"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("color_string", "color_16bit", "color_24bit", "color_preview")
    FUNCTION = "convert_color"
    CATEGORY = "image/color"

    def rgb_to_16bit(self, r, g, b, format='RGB565'):
        """
        Convert RGB values (0-255) to 16-bit color value
        
        Args:
            r, g, b: 8-bit color values (0-255)
            format: 'RGB565' or 'RGB555'
        
        Returns:
            16-bit color value
        """
        if format == 'RGB565':
            r5 = int(r * 31 / 255)
            g6 = int(g * 63 / 255)
            b5 = int(b * 31 / 255)
            return (r5 << 11) | (g6 << 5) | b5
        
        elif format == 'RGB555':
            r5 = int(r * 31 / 255)
            g5 = int(g * 31 / 255)
            b5 = int(b * 31 / 255)
            return (r5 << 10) | (g5 << 5) | b5
        
        else:
            raise ValueError("Format must be 'RGB565' or 'RGB555'")
    
    def rgb_to_24bit(self, r, g, b):
        """
        Convert RGB values (0-255) to 24-bit color value (0-16777215)
        This is compatible with ComfyUI's emptyImage node.
        
        Args:
            r, g, b: 8-bit color values (0-255)
        
        Returns:
            24-bit color value
        """
        return (r << 16) | (g << 8) | b
    
    def create_color_preview(self, r, g, b, size=128):
        """
        Create a preview image of the color
        
        Args:
            r, g, b: 8-bit color values (0-255)
            size: Size of the preview image in pixels
            
        Returns:
            Tensor representing the color preview image
        """
        # Create a solid color image
        img = Image.new('RGB', (size, size), (r, g, b))
        
        # Convert to numpy array and normalize to 0-1 range
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor with batch dimension
        return torch.from_numpy(img_np)[None, ]
    
    def convert_color(self, red, green, blue, format="RGB565"):
        """
        Convert RGB values to multiple color formats and preview image.
        
        Args:
            red, green, blue: 8-bit color values (0-255)
            format: 16-bit color format ('RGB565' or 'RGB555')
            
        Returns:
            Tuple of (color_string, color_16bit, color_24bit, color_preview)
        """
        # Generate the hex color string (e.g., "#FF0000" for red)
        color_string = f"#{red:02X}{green:02X}{blue:02X}"
        
        # Calculate the 16-bit color value
        color_16bit = self.rgb_to_16bit(red, green, blue, format)
        
        # Calculate the 24-bit color value (compatible with ComfyUI's emptyImage)
        color_24bit = self.rgb_to_24bit(red, green, blue)
        
        # Create a preview image of the color
        color_preview = self.create_color_preview(red, green, blue)
        
        return (color_string, color_16bit, color_24bit, color_preview)
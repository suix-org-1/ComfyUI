import torch
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO
from server import PromptServer
from comfy_execution.graph import ExecutionBlocker

class ImageColorSampler:
    """
    This node allows clicking on an input image to sample colors,
    creating a color palette from the selected sample points.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sample_points": ("STRING", {"default": "[]", "multiline": True}),
                "palette_size": ("INT", {"default": 128, "min": 32, "max": 512}),
                "sample_size": ("INT", {"default": 1, "min": 1, "max": 30}),
                "wait_for_input": ("BOOLEAN", {"default": True})
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("palette", "sampled_colors", "hex_codes", "swatches", "rgb_24bit", "rgb_565", "rgb_values")
    FUNCTION = "create_palette"
    CATEGORY = "image/color"
    
    # Enable dynamic outputs for individual colors
    OUTPUT_NODE = True
    
    # Enable list output for swatches, hex_codes, rgb_24bit, rgb_565, and rgb_values
    OUTPUT_IS_LIST = [False, False, True, True, True, True, True]
    
    # Track which nodes are waiting for user input
    waiting_nodes = set()
    
    def tensor_to_base64_image(self, tensor):
        """Convert a torch tensor to a base64 encoded image string"""
        # Convert tensor to numpy and then to PIL image
        img_np = tensor.cpu().numpy().squeeze(0)
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Save to a bytes buffer and convert to base64
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    def rgb_to_16bit(self, r, g, b, format='RGB565'):
        """
        Convert RGB values (0-255) to 16-bit color value in RGB565 format
        
        Args:
            r, g, b: 8-bit color values (0-255)
            format: Currently only 'RGB565' is supported
        
        Returns:
            16-bit color value
        """
        # Convert to RGB565 format (5 bits R, 6 bits G, 5 bits B)
        r5 = int(r * 31 / 255)
        g6 = int(g * 63 / 255)
        b5 = int(b * 31 / 255)
        return (r5 << 11) | (g6 << 5) | b5
    
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
        
    
    def create_palette(self, image, sample_points, palette_size=128, sample_size=5, wait_for_input=True, node_id=None):
        """
        Creates a color palette from the sampled points on the image.
        
        Args:
            image: Input image tensor
            sample_points: JSON string of sample points coordinates and colors
            palette_size: Size of the palette image (height in pixels)
            sample_size: Size of sample area (radius) for color averaging
            wait_for_input: Whether to block execution waiting for user input
            node_id: Unique ID of this node instance
        
        Returns:
            Tuple of (palette_image, sampled_colors_json, hex_codes_list, swatches_list, rgb_24bit_list, rgb_565_list, rgb_values_list)
        """
        # Parse the sample points
        try:
            points = json.loads(sample_points)
        except json.JSONDecodeError:
            points = []
        
        # Check if this is the initial call or a resumption after user input
        is_initial_call = node_id not in self.waiting_nodes
        
        # For initial call, send image data to the UI for interactive editing
        if (is_initial_call and wait_for_input):
            # Convert image tensor to a base64 string for sending to UI
            img_base64 = self.tensor_to_base64_image(image)
            
            # Send image and current points to the UI
            ui_data = {
                "image": img_base64,
                "sample_points": points,
                "sample_size": sample_size,
                "node_id": node_id
            }
            
            # Send message to UI to display the image for interaction
            PromptServer.instance.send_sync("image_sampler_init", {"node": node_id, "data": ui_data})
            
            # Add to waiting nodes and block execution
            self.waiting_nodes.add(node_id)
            
            # Return ExecutionBlocker for all outputs
            return (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))
        
        # Remove from waiting list if resuming
        if node_id in self.waiting_nodes:
            self.waiting_nodes.remove(node_id)
        
        # Convert image tensor to numpy array
        img_np = image.cpu().numpy().squeeze(0)
        
        # Image dimensions
        height, width, _ = img_np.shape
        
        # If no points, return empty palette
        if not points:
            # Create empty palette
            palette_img = np.zeros((palette_size, palette_size, 3), dtype=np.float32)
            palette_tensor = torch.from_numpy(palette_img)[None, ]
            empty_swatch = torch.from_numpy(np.zeros((palette_size, palette_size, 3), dtype=np.float32))[None, ]
            return (palette_tensor, "[]", [], [empty_swatch], [], [], [])
            
        # Calculate colors for each sample point
        sampled_colors = []
        hex_codes = []
        swatches = []
        rgb_24bit = []
        rgb_565 = []
        rgb_values = []
        
        for point in points:
            x = int(point["x"] * width)
            y = int(point["y"] * height)
            
            # Use exact color from JavaScript when sample_size is 1, otherwise do averaging
            if sample_size == 1 and "color" in point and isinstance(point["color"], str) and point["color"].startswith("#"):
                # Use the hex color directly from JavaScript for exact values
                hex_color = point["color"]
                
                # Parse hex color to RGB
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
            else:
                # Use averaging for larger sample sizes or when color isn't specified
                # Ensure coordinates are within bounds
                x = max(sample_size, min(width - sample_size - 1, x))
                y = max(sample_size, min(height - sample_size - 1, y))
                
                # Sample area - take average color in the sample radius
                sample_area = img_np[y-sample_size:y+sample_size+1, x-sample_size:x+sample_size+1]
                avg_color = np.mean(sample_area, axis=(0, 1))
                
                # Convert to 8-bit RGB
                r, g, b = [int(c * 255) for c in avg_color]
                
                # Create hex code
                hex_color = f"#{r:02X}{g:02X}{b:02X}"
            
            hex_codes.append(hex_color)
            
            # Add to colors list with position info
            sampled_colors.append({
                "position": {"x": point["x"], "y": point["y"]},
                "color": {"r": r, "g": g, "b": b},
                "hex": hex_color
            })
            
            # Create a swatch image for this color
            swatch_img = np.zeros((palette_size, palette_size, 3), dtype=np.float32)
            swatch_img[:, :, 0] = r / 255.0
            swatch_img[:, :, 1] = g / 255.0
            swatch_img[:, :, 2] = b / 255.0
            swatch_tensor = torch.from_numpy(swatch_img)[None, ]
            swatches.append(swatch_tensor)
            
            # Add 24-bit RGB value to list using the dedicated method
            rgb_24bit.append(self.rgb_to_24bit(r, g, b))
            
            # Add 16-bit RGB565 value to list
            rgb_565.append(self.rgb_to_16bit(r, g, b, 'RGB565'))
            
            # Add RGB values to list
            rgb_values.append(f"({r}, {g}, {b})")
            
        # Create palette image
        num_colors = len(sampled_colors)
        if num_colors == 0:
            # Create empty palette
            palette_img = np.zeros((palette_size, palette_size, 3), dtype=np.float32)
            palette_tensor = torch.from_numpy(palette_img)[None, ]
            empty_swatch = torch.from_numpy(np.zeros((palette_size, palette_size, 3), dtype=np.float32))[None, ]
            return (palette_tensor, "[]", [], [empty_swatch], [], [], [])
            
        # Create palette image - a horizontal strip of colors
        stripe_height = palette_size
        stripe_width = palette_size // num_colors if num_colors > 0 else palette_size
        
        palette_img = np.zeros((stripe_height, palette_size, 3), dtype=np.float32)
        
        for i, color_data in enumerate(sampled_colors):
            color = color_data["color"]
            start_x = i * stripe_width
            end_x = (i + 1) * stripe_width if i < num_colors - 1 else palette_size
            
            # Fill the stripe with the color
            palette_img[:, start_x:end_x, 0] = color["r"] / 255.0
            palette_img[:, start_x:end_x, 1] = color["g"] / 255.0
            palette_img[:, start_x:end_x, 2] = color["b"] / 255.0
        
        # Convert to tensor
        palette_tensor = torch.from_numpy(palette_img)[None, ]
        
        # Return outputs based on format preference
        json_colors = json.dumps(sampled_colors)
        
        # Prepare dynamic outputs (individual hex codes)
        self.output_colors = hex_codes
        
        # Return all outputs, including the swatches list, 24-bit RGB values, RGB565 values, and RGB values
        return (palette_tensor, json_colors, hex_codes, swatches, rgb_24bit, rgb_565, rgb_values)
    
    # Method to provide dynamic outputs for individual colors
    def get_output_for_node_type(self, node):
        outputs = {"ui": {"text": ""}}  # Default empty output
        
        # Check if we have output_colors
        if hasattr(self, "output_colors") and self.output_colors:
            # Create dynamic outputs for each color
            for i, hex_code in enumerate(self.output_colors):
                outputs[f"color_{i+1}"] = ("STRING", {"color": hex_code})
            
            # Add UI description showing how many colors are available
            outputs["ui"]["text"] = f"{len(self.output_colors)} colors sampled"
        else:
            outputs["ui"]["text"] = "No colors sampled yet"
            
        return outputs

# Add routes for handling the continuation of workflow
@PromptServer.instance.routes.post("/image_sampler/continue")
async def image_sampler_continue(request):
    """Handle when user is done selecting color samples and wants to continue"""
    data = await request.json()
    node_id = data.get("node_id")
    sample_points = data.get("sample_points", "[]")
    
    # Update the sample_points widget value to continue with new points
    PromptServer.instance.send_sync("image_sampler_update", {
        "node": node_id,
        "widget_name": "sample_points",
        "value": json.dumps(sample_points)
    })
    
    return {"status": "success"}
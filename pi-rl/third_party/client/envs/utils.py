import numpy as np
from openpi_client import image_tools

def process_image_for_obs(img, bgr_to_rgb=False, image_size=None):
    """Process image for observation: convert CHW to HWC, normalize to uint8, and resize.
    
    Args:
        img: Image array (can be in various formats)
        bgr_to_rgb: Whether to convert BGR to RGB (default: False)
        
    Returns:
        Processed image array in HWC uint8 format
    """
    # Convert CHW to HWC if needed
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    # Convert to uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert BGR to RGB if needed
    if bgr_to_rgb and img.shape[-1] == 3:
        img = img[..., ::-1]

    if image_size is not None:
        img = image_tools.resize_with_pad(img, image_size[0], image_size[1])

    return img
    
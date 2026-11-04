import numpy as np 
import cv2
import math
import random
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def normalizer(image, photometric, dcm_meta=None):
    image = image.astype(np.float32)

    if dcm_meta is not None:
        try:
            image = apply_voi_lut(image, dcm_meta)
        except Exception:
            pass  # missing/malformed VOI LUT tags -> fall back to raw values

    if photometric == 'MONOCHROME1':
        image = image.max() - image   # invert AFTER windowing

    image_min, image_max = np.min(image), np.max(image)
    if image_max - image_min > 0:
        image = (image - image_min) / (image_max - image_min) * 255.0
    else:
        image = np.zeros_like(image)
    return image.astype(np.uint8)

    
def get_anatomy_bbox(image, threshold_value=15):
    """
    Finds the bounding box of the main anatomy in a normalized 8-bit image
    by isolating the largest connected component.
    
    Returns:
        bbox (list): [x_min, y_min, x_max, y_max]
    """
    # 1. Light Gaussian blur to smooth out internal textures and noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Binarize the image
    # Any pixel brighter than threshold_value becomes 255 (white), rest become 0 (black)
    _, binary_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 3. Clean up the mask using Morphological Closing
    # This fills in small holes inside the anatomy (like dark spots in lungs or ventricles)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find all external contours in the mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fallback if image is completely black or no contours found
    if not contours:
        h, w = image.shape
        return [0, 0, w, h]
        
    # 5. Extract the largest contour by area (this drops small text/markers)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 6. Get the standard bounding box of that contour
    x_min, y_min, w, h = cv2.boundingRect(largest_contour)
    x_max = x_min + w
    y_max = y_min + h
    
    return [x_min, y_min, x_max, y_max]

def get_roi_random_crop(image, bbox, crop_h, crop_w, roi_padding=10, random_position=True):
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, x_min - roi_padding)
    y_min = max(0, y_min - roi_padding)
    x_max = min(w, x_max + roi_padding)
    y_max = min(h, y_max + roi_padding)
    bbox_w, bbox_h = x_max - x_min, y_max - y_min

    if bbox_w > crop_w or bbox_h > crop_h:
        scale = min(crop_w / bbox_w, crop_h / bbox_h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_min, y_min, x_max, y_max = [v * scale for v in (x_min, y_min, x_max, y_max)]
        h, w = new_h, new_w
        bbox_w, bbox_h = x_max - x_min, y_max - y_min

    pad_b, pad_r = max(0, crop_h - h), max(0, crop_w - w)
    if pad_b or pad_r:
        image = cv2.copyMakeBorder(image, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        h, w = image.shape[:2]

    x_lo, x_hi = max(0, x_max - crop_w), min(x_min, w - crop_w)
    y_lo, y_hi = max(0, y_max - crop_h), min(y_min, h - crop_h)
    x_lo, x_hi = math.ceil(x_lo), math.floor(x_hi)
    y_lo, y_hi = math.ceil(y_lo), math.floor(y_hi)
    x_hi, y_hi = max(x_hi, x_lo), max(y_hi, y_lo)

    x_lo, x_hi = min(x_lo, w - crop_w), min(x_hi, w - crop_w)
    y_lo, y_hi = min(y_lo, h - crop_h), min(y_hi, h - crop_h)
    x_lo, y_lo = max(x_lo, 0), max(y_lo, 0)

    if random_position:
        x_start, y_start = random.randint(x_lo, x_hi), random.randint(y_lo, y_hi)
    else:
        x_start, y_start = (x_lo + x_hi) // 2, (y_lo + y_hi) // 2

    cropped = image[y_start:y_start + crop_h, x_start:x_start + crop_w]

    ch, cw = cropped.shape[:2]
    if ch != crop_h or cw != crop_w:
        cropped = cv2.copyMakeBorder(
            cropped, 0, max(0, crop_h - ch), 0, max(0, crop_w - cw),
            cv2.BORDER_CONSTANT, value=0
        )[:crop_h, :crop_w]

    return cropped


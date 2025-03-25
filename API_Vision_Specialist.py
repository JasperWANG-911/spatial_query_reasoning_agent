"""
API for Vision Specialist(SAM2).
This module provides a fixed SAM2 class for image segmentation and pixel measurements.
"""

import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor


class SAM2:
    """
    SAM2 class with functionality: 
        -(1) segmentation 
        -(2) generate smallest bbox for mask
        -(3) calculate width of bbox
        -(4) calculate height of bbox
        -(5) calculate area of bbox
    """
    def __init__(self, checkpoint_path='../SPATIAL_QUERY_REASONING_AGENT/sam_vit_h_4b8939.pth', model_type="vit_h"):
        """Initialize SAM2 model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def load_image(self, image_path):
        """ Load image and convert to RGB format """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Image not found at {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def set_image(self, image):
        """ Set image for prediction """
        self.predictor.set_image(image)
    
    def predict_mask(self, x1, y1, x2, y2):
        """ Segment object and return mask """
        input_point = np.array([[(x1 + x2) // 2, (y1 + y2) // 2]])  # Find center of bounding box
        input_label = np.array([1])

        # Segment object
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        if masks is None or len(masks) == 0:
            raise ValueError("Error: No masks were generated. Check your input bounding box or model.")

        return masks[np.argmax(scores)]  # Return mask with highest score

    def get_smallest_bounding_box(self, mask):
        """ Obtain smallest bounding box from mask """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Warning: No contours found in mask.")
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return np.int0(box), rect  # Convert coordinates to integer values

    def measure_width(self, image_path, x1, y1, x2, y2):
        """ Segment object and measure width """
        try:
            image = self.load_image(image_path)
            self.set_image(image)
            
            mask = self.predict_mask(x1, y1, x2, y2)
            
            box, rect = self.get_bounding_box(mask)
            
            self.visualize_segmentation(image, mask, box, rect)
            
            if rect:
                return min(rect[1])  # Take minimum of width and height as width
            return None
        
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None
        
    def measure_height(self, image_path, x1, y1, x2, y2):
        """ Segment object and measure width """
        try:
            image = self.load_image(image_path)
            self.set_image(image)
            
            mask = self.predict_mask(x1, y1, x2, y2)
            
            box, rect = self.get_bounding_box(mask)
            
            self.visualize_segmentation(image, mask, box, rect)
            
            if rect:
                return max(rect[1])  # Take maximum of width and height as height
            return None
        
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None
        
    def compute_mask_pixel(self, mask):
        """ calculate the number of pixels in the mask """
        pixel_count = np.sum(mask > 0) 
        return pixel_count 
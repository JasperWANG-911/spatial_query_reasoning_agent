import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


class SAM2:
    """
    SAM2 class with functionality: 
        -(1) segmentation 
        -(2) generate smallest bbox for mask
        -(3) calculate width of bbox
        -(4) calculate height of bbox
        -(5) calculate area of bbox
        -(6) visualize segmentation with measurements
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
            
            box, rect = self.get_smallest_bounding_box(mask)
            
            self.visualize_segmentation(image, mask, box, rect)
            
            if rect:
                return min(rect[1])  # Take minimum of width and height as width
            return None
        
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None
        
    def measure_height(self, image_path, x1, y1, x2, y2):
        """ Segment object and measure height """
        try:
            image = self.load_image(image_path)
            self.set_image(image)
            
            mask = self.predict_mask(x1, y1, x2, y2)
            
            box, rect = self.get_smallest_bounding_box(mask)
            
            self.visualize_segmentation(image, mask, box, rect)
            
            if rect:
                return max(rect[1])  # Take maximum of width and height as height
            return None
        
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None
        
    def compute_mask_pixel(self, mask):
        """ Calculate the number of pixels in the mask """
        pixel_count = np.sum(mask > 0) 
        return pixel_count
    
    def visualize_segmentation(self, image, mask, box=None, rect=None, save_path=None, obj_name=None):
        """
        Visualize segmentation results with measurements.
        
        Args:
            image: Input image
            mask: Segmentation mask
            box: Bounding box points
            rect: Rectangle information from cv2.minAreaRect
            save_path: Path to save visualization (if None, display instead)
            obj_name: Optional object name to display
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Create a colored overlay for the mask
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[mask > 0] = [255, 0, 0]  # Red color for the mask
        
        # Apply the mask as a semi-transparent overlay
        alpha = 0.4
        vis_image = cv2.addWeighted(color_mask, alpha, vis_image, 1 - alpha, 0)
        
        # Draw the oriented bounding box if provided
        if box is not None and box.shape[0] == 4:
            # Draw the oriented bounding box
            for i in range(4):
                cv2.line(vis_image, tuple(box[i]), tuple(box[(i+1) % 4]), (0, 255, 0), 2)
        
        # Calculate and display dimensions
        if rect is not None:
            # Get box center, width, height, and angle
            center, dimensions, angle = rect
            width, height = dimensions
            
            # Ensure width is the shorter dimension
            width, height = min(dimensions), max(dimensions)
            
            # Calculate the center of the box
            center_x, center_y = int(center[0]), int(center[1])
            
            # Display dimensions with background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Create text
            text = f"W: {int(width)}px H: {int(height)}px"
            if obj_name:
                text = f"{obj_name}: {text}"
            
            # Get text size for background box
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            
            # Draw semi-transparent background for text
            overlay = vis_image.copy()
            
            # Position text box at top-right of the object
            text_x = center_x + int(width/2)
            text_y = center_y - int(height/2) - 10
            
            # Ensure text box is within image bounds
            if text_x + text_w > vis_image.shape[1]:
                text_x = vis_image.shape[1] - text_w - 10
            if text_y - text_h < 0:
                text_y = text_h + 10
            
            # Draw background rectangle
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_h - 5), 
                         (text_x + text_w + 5, text_y + 5),
                         (255, 100, 100), -1)
            
            # Apply transparency
            cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
            
            # Add text
            cv2.putText(vis_image, text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Save or display the visualization
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to {save_path}")
        else:
            plt.figure(figsize=(12, 12))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return vis_image
    
    def visualize_multiple_objects(self, image_path, objects_info, save_path=None):
        """
        Visualize multiple segmented objects in one image.
        
        Args:
            image_path: Path to the input image
            objects_info: List of dictionaries with 'obj_name', 'mask', 'bbox', etc.
            save_path: Path to save visualization (if None, display instead)
        """
        # Load the original image
        image = self.load_image(image_path)
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Generate random colors for different objects
        import random
        
        # Create a colored overlay for each mask
        for i, obj_info in enumerate(objects_info):
            if 'mask' not in obj_info:
                continue
                
            mask = obj_info['mask']
            obj_name = obj_info.get('obj', f"Object_{i+1}")
            
            # Generate a random color for this object
            color = [random.randint(0, 255) for _ in range(3)]
            
            # Create colored mask and apply as overlay
            color_mask = np.zeros_like(image, dtype=np.uint8)
            color_mask[mask > 0] = color
            
            # Apply as semi-transparent overlay
            alpha = 0.4
            vis_image = cv2.addWeighted(color_mask, alpha, vis_image, 1, 0)
            
            # Get bounding box info
            if 'bbox' in obj_info:
                x1, y1, x2, y2 = obj_info['bbox']
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Display dimensions
            width_pixel = obj_info.get('width_pixel', 0)
            height_pixel = obj_info.get('height_pixel', 0)
            
            if width_pixel > 0 and height_pixel > 0:
                # Find center of the object (using mask centroid)
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # Create text
                    text = f"{obj_name}: W: {int(width_pixel)}px H: {int(height_pixel)}px"
                    
                    # Get text size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_w, text_h = text_size
                    
                    # Position text
                    text_x = center_x
                    text_y = center_y - 10
                    
                    # Ensure text box is within image bounds
                    if text_x + text_w > vis_image.shape[1]:
                        text_x = vis_image.shape[1] - text_w - 10
                    if text_y - text_h < 0:
                        text_y = text_h + 10
                    
                    # Draw background rectangle
                    overlay = vis_image.copy()
                    cv2.rectangle(overlay, 
                                 (text_x - 5, text_y - text_h - 5), 
                                 (text_x + text_w + 5, text_y + 5),
                                 color, -1)
                    
                    # Apply transparency
                    cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
                    
                    # Add text
                    cv2.putText(vis_image, text, (text_x, text_y), 
                               font, font_scale, (255, 255, 255), thickness)
        
        # Save or display the visualization
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"Multi-object visualization saved to {save_path}")
        else:
            plt.figure(figsize=(15, 15))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return vis_image
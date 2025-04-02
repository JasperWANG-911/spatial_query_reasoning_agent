"""
Agent for real-world spatial metric calculation.
This module calculates scale factors from reference objects detected by the Reference Detection Agent.
"""

import numpy as np
from typing import Dict, Any

class SpatialMetricAgent:
    """
    Agent for calculating real-world spatial metrics from detected reference objects.
    Implements a weighted confidence approach where objects with larger pixel areas receive higher weights.
    """
    
    def run(self, reference_objects: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the scale factor from reference objects.
        
        Args:
            reference_objects: Dictionary of reference objects with their measurements
        
        Returns:
            Dictionary with scale factor, confidence, and analysis
        """
        # Filter objects with real-world dimensions
        valid_objects = {k: v for k, v in reference_objects.items() 
                        if ('width_m' in v and 'width_pixel' in v) or 
                           ('length_m' in v and 'height_pixel' in v) or 
                           ('area_m' in v and 'area_pixel' in v)}
        
        if not valid_objects:
            return {"status": "error", "message": "No objects with known dimensions found", "scale_factor": None}
        
        # Calculate individual scales and weights
        scales = []
        weights = []
        individual_scales = {}
        
        # Calculate total pixel area for weight normalization
        total_area = sum(obj.get('area_pixel', 0) for obj in valid_objects.values())
        
        for obj_name, obj in valid_objects.items():
            obj_scales = {}
            
            # Calculate scale factors from different dimensions
            if 'width_m' in obj and 'width_pixel' in obj and obj['width_pixel'] > 0:
                width_scale = obj['width_m'] / obj['width_pixel']
                obj_scales['width_scale'] = width_scale
                scales.append(width_scale)
                weights.append(obj.get('area_pixel', 0))
            
            if 'length_m' in obj and 'height_pixel' in obj and obj['height_pixel'] > 0:
                length_scale = obj['length_m'] / obj['height_pixel']
                obj_scales['length_scale'] = length_scale
                scales.append(length_scale)
                weights.append(obj.get('area_pixel', 0))
            
            if 'area_m' in obj and 'area_pixel' in obj and obj['area_pixel'] > 0:
                area_scale = np.sqrt(obj['area_m'] / obj['area_pixel'])
                obj_scales['area_scale'] = area_scale
                scales.append(area_scale)
                weights.append(obj.get('area_pixel', 0))
            
            # Store individual scales if any were calculated
            if obj_scales:
                individual_scales[obj_name] = obj_scales
        
        if not scales:
            return {"status": "error", "message": "Could not calculate any scales", "scale_factor": None}
        
        # Convert to numpy arrays
        scales_array = np.array(scales)
        weights_array = np.array(weights)
        
        # Normalize weights
        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones_like(scales_array) / len(scales_array)
        
        # Calculate weighted scale and statistics
        weighted_scale = np.sum(scales_array * weights_array)
        mean_scale = np.mean(scales_array)
        std_dev = np.std(scales_array)
        
        # Calculate confidence based on coefficient of variation
        confidence = max(0, min(1, 1 - (std_dev / mean_scale))) if mean_scale > 0 else 0
        
        # Create result dictionary
        result = {
            "status": "success",
            "scale_factor": weighted_scale,
            "confidence": confidence,
            "statistics": {
                "mean": mean_scale,
                "std_dev": std_dev,
                "count": len(scales)
            }
        }
        
        return result
    
    def apply_scale_to_measurement(self, pixels: float, scale_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a pixel measurement to real-world units.
        
        Args:
            pixels: Measurement in pixels
            scale_result: Result from run method
            
        Returns:
            Dictionary with the measurement in real-world units
        """
        if scale_result["status"] != "success":
            return {"status": "error", "message": "Invalid scale factor", "measurement": None}
        
        scale_factor = scale_result["scale_factor"]
        confidence = scale_result.get("confidence", 0)
        
        # Convert to real-world units
        real_world = pixels * scale_factor
        
        # Calculate error bounds
        std_dev = scale_result["statistics"]["std_dev"]
        error_margin = std_dev / scale_factor if scale_factor > 0 else 0
        
        return {
            "status": "success",
            "pixels": pixels,
            "meters": real_world,
            "confidence": confidence,
            "error_margin_percent": error_margin * 100,
            "range": [real_world * (1 - error_margin), real_world * (1 + error_margin)]
        }


# Example usage
if __name__ == "__main__":
    # Sample reference objects
    reference_objects = {
  "tennis_court_1": {
    "width_pixel": 87.0731430053711,
    "height_pixel": 186.51870727539062,
    "area_pixel": 15709
  },
  "tennis_court_2": {
    "width_pixel": 86.89117431640625,
    "height_pixel": 187.02455139160156,
    "area_pixel": 16082
  },
  "tennis_court_3": {
    "width_pixel": 86.7344970703125,
    "height_pixel": 186.53628540039062,
    "area_pixel": 15987
  },
  "tennis_court_4": {
    "width_pixel": 87.48783874511719,
    "height_pixel": 186.18226623535156,
    "area_pixel": 15743
  },
  "tennis_court_5": {
    "width_pixel": 86.74920654296875,
    "height_pixel": 187.03335571289062,
    "area_pixel": 15838
  },
  "tennis_court_6": {
    "width_pixel": 85.97868347167969,
    "height_pixel": 186.3462677001953,
    "area_pixel": 15866
  },
  "tennis_court_7": {
    "width_pixel": 85.7127685546875,
    "height_pixel": 185.6812286376953,
    "area_pixel": 15798
  },
  "tennis_court_8": {
    "width_pixel": 85.43696594238281,
    "height_pixel": 185.66976928710938,
    "area_pixel": 15577
  },
  "tennis_court_9": {
    "width_pixel": 85.4000015258789,
    "height_pixel": 185.60000610351562,
    "area_pixel": 15695
  },
  "tennis_court_10": {
    "width_pixel": 85.7127685546875,
    "height_pixel": 185.6812286376953,
    "area_pixel": 15798
  },
  "tennis_court_11": {
    "width_pixel": 86.57191467285156,
    "height_pixel": 186.53997802734375,
    "area_pixel": 15971
  }
}
    
    # Calculate scale
    agent = SpatialMetricAgent()
    result = agent.run(reference_objects)
    
    # Print results
    import json
    print(json.dumps(result, indent=2))
    
    # Example application
    measurement = agent.apply_scale_to_measurement(100, result)
    print(f"\n100 pixels = {measurement['meters']:.2f} meters (±{measurement['error_margin_percent']:.1f}%)")
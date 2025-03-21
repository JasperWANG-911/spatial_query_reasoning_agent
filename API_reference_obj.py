import os
import json
import base64
import numpy as np
import math

'''
All functions below need to be implemented as instructed:
(1) call_geochat_for_image_caption: add a method to call geochat for generating caption
(2) select_optimal_reference: find a suitable model for this(GPT4?), which can summarise and filter key reference objects from the caption
(3) call_geochat_for_segmentation: add a method to call geochat for segmentation
(4) call_unidepth: add a method call unidepth for depth analysis
.....
More functions will be added to this list
'''


def call_geochat_for_image_caption(image_path: str) -> str:
    """
    Use GeoChat to generate a text caption for image
    """
    
    # TBC：geochat是否支持system prompt?
    system_prompt = """You are an expert in analyzing remote sensing images. 
    Your task is to provide a comprehensive caption for the given remote sensing image.
    
    Include information about:
    1. The general scene type (urban, rural, coastal, etc.)
    2. Major visible features (buildings, roads, natural features, etc.)
    3. Approximate spatial arrangement of elements
    4. Any notable objects with standardized sizes (vehicles, sports fields, etc.)
    5. Potential scale indicators in the image
    
    Provide a clear, concise description without speculation. Focus on what is visibly present."""

def generate_reference_list():

def select_optimal_references(api_key: str, reference_objects: List[Dict], target_metric: str) -> List[Dict]:
    """
    Prioritize reference objects based on their reliability for the specific measurement task
    """
    
    # Generate code to filter and rank reference objects
    system_prompt = """You are an assistant that selects the most reliable reference objects for scale estimation.
    
    Given a list of potential reference objects and a target metric to measure (area, length, distance, or perimeter),
    your task is to return a JSON response with the most reliable reference objects to use for scale estimation.
    
    Consider the following criteria:
    1. For area measurements, select objects with well-defined areas
    2. For length measurements, select linear objects of known length
    3. For distance measurements, select objects of known dimensions near the measurement points
    4. Prefer objects with higher confidence scores
    5. Prefer objects with standardized sizes (cars, courts, pools, fields)
    6. Avoid objects that might have significant size variations
    7. Select 2-3 reliable references rather than many uncertain ones
    
    Return a JSON list of selected reference objects with their known real-world dimensions."""

def call_geochat_for_segmentation():
    return True

def calculate_pixel_metrics(segmentation_results: Dict, target_metric: str) -> Dict:
    """
    Calculate pixel-level metrics (area, length, perimeter, etc.) for all segmented objects
    """
    
    pixel_metrics = {
        "reference_metrics": [],
        "target_metrics": {}
    }
    
    # Calculate metrics for reference objects
    for ref_segment in segmentation_results["reference_segments"]:
        mask = np.array(ref_segment["mask"])
        bbox = ref_segment["bbox"]
        
        # Calculate pixel area
        pixel_area = np.sum(mask)
        
        # Calculate approximate perimeter (in a real implementation, this would be more precise)
        # For simplicity, we're just using the bounding box perimeter
        pixel_perimeter = 2 * ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]))
        
        # For length metrics (if applicable)
        pixel_length = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1]))
        
        pixel_metrics["reference_metrics"].append({
            "object_id": ref_segment["object_id"],
            "object_type": ref_segment["object_type"],
            "pixel_area": float(pixel_area),
            "pixel_perimeter": float(pixel_perimeter),
            "pixel_length": float(pixel_length),
            "real_world_dimensions": ref_segment["real_world_dimensions"]
        })
    
    # Calculate metrics for target object
    if segmentation_results.get("target_segment"):
        target_mask = np.array(segmentation_results["target_segment"]["mask"])
        target_bbox = segmentation_results["target_segment"]["bbox"]
        
        # Calculate pixel area
        target_pixel_area = np.sum(target_mask)
        
        # Calculate approximate perimeter
        target_pixel_perimeter = 2 * ((target_bbox[2] - target_bbox[0]) + (target_bbox[3] - target_bbox[1]))
        
        # For length metrics
        target_pixel_length = max((target_bbox[2] - target_bbox[0]), (target_bbox[3] - target_bbox[1]))
        
        pixel_metrics["target_metrics"] = {
            "pixel_area": float(target_pixel_area),
            "pixel_perimeter": float(target_pixel_perimeter),
            "pixel_length": float(target_pixel_length)
        }
    
    print(f"✅ Calculated pixel metrics for {len(pixel_metrics['reference_metrics'])} reference objects and target object")
    return pixel_metrics

def call_unidepth():
    'import unidepth model'
    return True

def infer_scale_factor(selected_references: List[Dict], pixel_metrics: Dict) -> Dict:
    """
    Infer the scale factor based on reference objects using weighted approach
    """
    
    scale_factors = []
    weights = []
    
    # Match reference objects with their pixel metrics
    for ref in selected_references:
        ref_id = ref.get("id", "unknown")
        
        # Find matching pixel metrics
        matching_metrics = next((m for m in pixel_metrics["reference_metrics"] if m["object_id"] == ref_id), None)
        
        if not matching_metrics:
            continue
            
        # Get real-world dimensions
        real_world_dims = matching_metrics.get("real_world_dimensions", {})
        
        # Calculate scale factor based on the type of object and available dimensions
        object_type = ref.get("type", "").lower()
        pixel_area = matching_metrics.get("pixel_area", 0)
        pixel_length = matching_metrics.get("pixel_length", 0)
        
        if "area" in real_world_dims and pixel_area > 0:
            # For area reference (e.g., courts, pools, fields)
            real_world_area = real_world_dims["area"]
            scale_factor = math.sqrt(real_world_area / pixel_area)  # Meters per pixel
            weight = pixel_area  # Larger objects get higher weight
            
            scale_factors.append(scale_factor)
            weights.append(weight)
            
        elif "length" in real_world_dims and pixel_length > 0:
            # For length reference (e.g., cars, road lanes)
            real_world_length = real_world_dims["length"]
            scale_factor = real_world_length / pixel_length  # Meters per pixel
            weight = pixel_length  # Longer objects get higher weight
            
            scale_factors.append(scale_factor)
            weights.append(weight)
            
        elif "width" in real_world_dims and "height" in real_world_dims:
            # For objects with width and height
            real_world_width = real_world_dims["width"]
            real_world_height = real_world_dims["height"]
            pixel_width = matching_metrics["bbox"][2] - matching_metrics["bbox"][0]
            pixel_height = matching_metrics["bbox"][3] - matching_metrics["bbox"][1]
            
            scale_factor_width = real_world_width / pixel_width
            scale_factor_height = real_world_height / pixel_height
            
            # Average of width and height scale factors
            scale_factor = (scale_factor_width + scale_factor_height) / 2
            weight = pixel_width * pixel_height  # Area as weight
            
            scale_factors.append(scale_factor)
            weights.append(weight)
    
    # Calculate weighted average scale factor
    if scale_factors and weights:
        total_weight = sum(weights)
        weighted_scale_factor = sum(sf * w for sf, w in zip(scale_factors, weights)) / total_weight
        
        # Calculate confidence based on variance of scale factors
        if len(scale_factors) > 1:
            variance = sum((sf - weighted_scale_factor) ** 2 * w for sf, w in zip(scale_factors, weights)) / total_weight
            std_dev = math.sqrt(variance)
            confidence = 1 - min(1, std_dev / weighted_scale_factor)
        else:
            confidence = 0.5  # Medium confidence if only one reference
        
        result = {
            "scale_factor": weighted_scale_factor,  # Meters per pixel
            "confidence": confidence,
            "individual_scale_factors": [
                {"scale_factor": sf, "weight": w} for sf, w in zip(scale_factors, weights)
            ]
        }
        
        print(f"Inferred scale factor: {weighted_scale_factor:.6f} meters/pixel with {confidence:.2f} confidence")
        return result
    else:
        print("Could not infer scale factor - no valid reference objects")
        return {"scale_factor": 1.0, "confidence": 0.0, "individual_scale_factors": []}

def calculate_real_world_metrics(pixel_metrics: Dict, scale_factor: Dict, target_metric: str) -> Dict:
    """
    Calculate real-world measurements based on pixel metrics and scale factor
    """
    
    real_world_metrics = {}
    
    # Extract target pixel metrics
    target_metrics = pixel_metrics.get("target_metrics", {})
    
    # Get scale factor value
    sf = scale_factor.get("scale_factor", 1.0)
    confidence = scale_factor.get("confidence", 0.0)
    
    # Calculate real-world metrics based on target metric type
    if target_metric.lower() == "area":
        # Area calculation (scaling factor squared for area)
        pixel_area = target_metrics.get("pixel_area", 0)
        real_world_area = pixel_area * (sf ** 2)
        real_world_metrics["area"] = real_world_area
        real_world_metrics["unit"] = "square meters"
        
    elif target_metric.lower() == "length":
        # Length calculation
        pixel_length = target_metrics.get("pixel_length", 0)
        real_world_length = pixel_length * sf
        real_world_metrics["length"] = real_world_length
        real_world_metrics["unit"] = "meters"
        
    elif target_metric.lower() == "perimeter":
        # Perimeter calculation
        pixel_perimeter = target_metrics.get("pixel_perimeter", 0)
        real_world_perimeter = pixel_perimeter * sf
        real_world_metrics["perimeter"] = real_world_perimeter
        real_world_metrics["unit"] = "meters"
        
    elif target_metric.lower() == "distance":
        # Distance calculation (placeholder - in a real implementation, this would use two points)
        pixel_length = target_metrics.get("pixel_length", 0)
        real_world_distance = pixel_length * sf
        real_world_metrics["distance"] = real_world_distance
        real_world_metrics["unit"] = "meters"
        
    # Add confidence
    real_world_metrics["confidence"] = confidence
    
    return real_world_metrics


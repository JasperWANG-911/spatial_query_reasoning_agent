"""
API for Reference Object Selection (Core Version).
This module provides essential functions for reference object selection and quality assessment.
The agent is encouraged to develop additional custom functions as needed for specific scenarios.
"""

from typing import List, Dict, Any, Optional


def check_scale_consistency(reference_objects: List[Dict[str, Any]], tolerance: float = 0.3) -> List[Dict[str, Any]]:
    """
    Check if the scale ratios among objects of the same type are consistent,
    and filter out objects with inconsistent scales.
    
    Args:
        reference_objects: List of reference objects
        tolerance: Allowed scale variation tolerance (percentage)
        
    Returns:
        Filtered list of reference objects with consistent scales
    """
    # Group objects by type
    object_groups = {}
    for obj in reference_objects:
        obj_type = obj.get('obj', '').split('_')[0]  # Extract base type (e.g., 'car' from 'car_1')
        if obj_type not in object_groups:
            object_groups[obj_type] = []
        object_groups[obj_type].append(obj)
    
    # For each type, calculate scale ratios
    consistent_objects = []
    for obj_type, objects in object_groups.items():
        if len(objects) <= 1:
            consistent_objects.extend(objects)
            continue
        
        # Calculate width and length scales for each object (if data available)
        scales = []
        for obj in objects:
            if ('width_m' in obj and 'width_pixel' in obj and 
                obj['width_m'] is not None and obj['width_pixel'] > 0):
                scales.append(obj['width_m'] / obj['width_pixel'])
            
            if ('length_m' in obj and 'height_pixel' in obj and 
                obj['length_m'] is not None and obj['height_pixel'] > 0):
                scales.append(obj['length_m'] / obj['height_pixel'])
        
        if not scales:
            consistent_objects.extend(objects)
            continue
        
        # Calculate mean scale
        mean_scale = sum(scales) / len(scales)
        
        # Filter objects within tolerance range
        filtered_objects = []
        for obj in objects:
            obj_scales = []
            if ('width_m' in obj and 'width_pixel' in obj and 
                obj['width_m'] is not None and obj['width_pixel'] > 0):
                obj_scales.append(obj['width_m'] / obj['width_pixel'])
            
            if ('length_m' in obj and 'height_pixel' in obj and 
                obj['length_m'] is not None and obj['height_pixel'] > 0):
                obj_scales.append(obj['length_m'] / obj['height_pixel'])
            
            if not obj_scales:
                continue
            
            obj_mean_scale = sum(obj_scales) / len(obj_scales)
            
            # Check if object's scale is within tolerance range
            if abs(obj_mean_scale - mean_scale) / mean_scale <= tolerance:
                filtered_objects.append(obj)
        
        consistent_objects.extend(filtered_objects)
    
    return consistent_objects


def estimate_object_quality(obj: Dict[str, Any]) -> float:
    """
    Estimate the quality of a reference object based on available metrics.
    Higher score means better quality for reference.
    
    Args:
        obj: Reference object
        
    Returns:
        Quality score between 0 and 1
    """
    score = 0.5  # Default score
    
    # Size preference: larger objects tend to have more reliable measurements
    area_pixel = obj.get('area_pixel', 0)
    if area_pixel > 10000:  # Very large object
        score += 0.3
    elif area_pixel > 1000:  # Medium size object
        score += 0.2
    elif area_pixel > 100:  # Small object
        score += 0.1
    
    # Aspect ratio: extreme aspect ratios might indicate poor segmentation
    width = obj.get('width_pixel', 1)
    height = obj.get('height_pixel', 1)
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 10
    
    if 1 <= aspect_ratio <= 3:  # Reasonable aspect ratio
        score += 0.1
    elif aspect_ratio > 10:  # Extreme aspect ratio
        score -= 0.2
    
    # Predefined reliability if available
    if 'reliability' in obj:
        score = 0.3 * score + 0.7 * obj['reliability']  # Weight reliability higher
    
    # Known dimensions boost
    has_real_dimensions = 'width_m' in obj or 'length_m' in obj or 'area_m' in obj
    if has_real_dimensions:
        score += 0.1
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))


def select_diverse_objects(reference_objects: List[Dict[str, Any]], min_objects: int = 3, keep_best_per_type: bool = True) -> List[Dict[str, Any]]:
    """
    Select a diverse set of reference objects, prioritizing different types but 
    keeping high-quality objects of the same type when needed.
    
    Args:
        reference_objects: List of reference objects
        min_objects: Minimum number of objects to select
        keep_best_per_type: Whether to keep the best object of each type before considering others
        
    Returns:
        List of diverse reference objects
    """
    if len(reference_objects) <= min_objects:
        return reference_objects
    
    # Group objects by type
    object_groups = {}
    for obj in reference_objects:
        obj_type = obj.get('obj', '').split('_')[0]  # Extract base type
        if obj_type not in object_groups:
            object_groups[obj_type] = []
        object_groups[obj_type].append(obj)
    
    # First pass: select the best object of each type if requested
    selected = []
    remaining = []
    
    if keep_best_per_type:
        for obj_type, objects in object_groups.items():
            # Sort by quality and take the best
            sorted_objects = sorted(objects, key=lambda x: x.get('quality', 0), reverse=True)
            selected.append(sorted_objects[0])
            # Add remaining objects to the pool
            remaining.extend(sorted_objects[1:])
    else:
        # Just add all objects to the remaining pool
        for objects in object_groups.values():
            remaining.extend(objects)
    
    # If we need more objects, add from remaining pool based on quality and diversity score
    if len(selected) < min_objects and remaining:
        # Sort remaining by quality
        remaining = sorted(remaining, key=lambda x: x.get('quality', 0), reverse=True)
        
        # Add objects until we reach the minimum or run out
        while len(selected) < min_objects and remaining:
            selected.append(remaining.pop(0))
    
    return selected
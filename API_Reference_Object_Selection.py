"""
API for Reference Object Selection (Simplified Version).
This module provides core functions for filtering and prioritizing reference objects.
To fine-tune this model, we can add more pre-defined functions here to achieve a better analyse of reference object list.
"""

from typing import List, Dict, Any, Optional


def filter_by_area(reference_objects: List[Dict[str, Any]], min_area: int = 0, max_area: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Filter reference objects based on their area in pixels.
    
    Args:
        reference_objects: List of reference objects
        min_area: Minimum area in pixels (default: 0)
        max_area: Maximum area in pixels (default: None - no upper limit)
        
    Returns:
        Filtered list of reference objects
    """
    if max_area is None:
        return [obj for obj in reference_objects if obj.get('area_pixel', 0) >= min_area]
    else:
        return [obj for obj in reference_objects if min_area <= obj.get('area_pixel', 0) <= max_area]

def prioritize_by_size(reference_objects: List[Dict[str, Any]], prefer_larger: bool = True) -> List[Dict[str, Any]]:
    """
    Sort reference objects by their size (area).
    
    Args:
        reference_objects: List of reference objects
        prefer_larger: If True, prefer larger objects; otherwise prefer smaller ones
        
    Returns:
        Sorted list of reference objects
    """
    if prefer_larger:
        return sorted(reference_objects, key=lambda x: x.get('area_pixel', 0), reverse=True)
    else:
        return sorted(reference_objects, key=lambda x: x.get('area_pixel', 0))

def filter_by_known_dimensions(reference_objects: List[Dict[str, Any]], reference_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter reference objects to only include those with known real-world dimensions.
    
    Args:
        reference_objects: List of reference objects
        reference_list: List of reference objects with known dimensions
        
    Returns:
        Filtered list of reference objects
    """
    known_types = [ref['type'] for ref in reference_list]
    
    return [
        obj for obj in reference_objects 
        if any(known_type in obj.get('obj', '').lower().split('_')[0] for known_type in known_types)
    ]

def select_diverse_references(reference_objects: List[Dict[str, Any]], min_objects: int = 3) -> List[Dict[str, Any]]:
    """
    Select a diverse set of reference objects, preferring different types.
    
    Args:
        reference_objects: List of reference objects
        min_objects: Minimum number of objects to select
        
    Returns:
        List of diverse reference objects
    """
    selected = []
    selected_types = set()
    
    # First pass: select one of each type
    for obj in reference_objects:
        obj_type = obj.get('obj', '').split('_')[0]  # Extract base type
        
        if obj_type not in selected_types:
            selected.append(obj)
            selected_types.add(obj_type)
    
    # Second pass: add more objects if needed
    if len(selected) < min_objects:
        remaining = [obj for obj in reference_objects if obj not in selected]
        # Sort remaining by area (prefer larger objects)
        remaining = sorted(remaining, key=lambda x: x.get('area_pixel', 0), reverse=True)
        selected.extend(remaining[:min_objects - len(selected)])
    
    return selected

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
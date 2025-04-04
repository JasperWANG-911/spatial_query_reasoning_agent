"""
API_Algorithm.py - Core algorithms for scale factor calculation in remote sensing imagery.

This module provides three fundamental algorithms to calculate scale factors (meters/pixel) 
from reference objects with known real-world dimensions.
"""

from typing import Dict, Any, List
import numpy as np
import math


def calculate_scale_by_weighted_area(reference_objects: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculate scale factor using a weighted approach based on pixel area.
    
    This method weights each reference object's scale factor by its pixel area,
    giving more influence to larger objects which typically have more reliable measurements.
    
    Formula:
        S = ∑(Wi * Si) where Wi = Ai / ∑Aj and Si = sqrt(area_m / area_pixel)
    
    Args:
        reference_objects: Dictionary of reference objects with their measurements
        
    Returns:
        Weighted scale factor in meters/pixel
    """
    if not reference_objects:
        raise ValueError("No reference objects provided")
    
    # Calculate individual scale factors and areas
    total_area_pixel = 0
    scale_factors = []
    weights = []
    
    for obj_name, obj_data in reference_objects.items():
        # Extract measurements
        area_pixel = obj_data.get('area_pixel', 0)
        area_m = obj_data.get('area_m', 0)
        
        # Skip objects with invalid measurements
        if area_pixel <= 0 or area_m <= 0:
            continue
            
        # Calculate scale factor for this object (meters/pixel)
        scale_factor = math.sqrt(area_m / area_pixel)
        
        scale_factors.append(scale_factor)
        weights.append(area_pixel)
        total_area_pixel += area_pixel
    
    # If no valid objects found, return error
    if not scale_factors:
        raise ValueError("No valid reference objects with area measurements")
    
    # Normalize weights
    if total_area_pixel > 0:
        weights = [w / total_area_pixel for w in weights]
    else:
        # Equal weights if total area is zero (shouldn't happen but just in case)
        weights = [1.0 / len(scale_factors) for _ in scale_factors]
    
    # Calculate weighted average
    weighted_scale = sum(w * s for w, s in zip(weights, scale_factors))
    
    return weighted_scale


def calculate_scale_by_least_squares(reference_objects: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculate scale factor using a least squares approach.
    
    This method minimizes the squared differences between real-world dimensions
    and pixel dimensions scaled by the factor, finding the optimal scaling.
    
    Args:
        reference_objects: Dictionary of reference objects with their measurements
        
    Returns:
        Least squares optimized scale factor in meters/pixel
    """
    if not reference_objects:
        raise ValueError("No reference objects provided")
    
    # Collect all dimension pairs (pixel and real-world)
    pixel_dims = []
    real_dims = []
    
    for obj_name, obj_data in reference_objects.items():
        # Add width dimension if available
        if obj_data.get('width_pixel', 0) > 0 and obj_data.get('width_m', 0) > 0:
            pixel_dims.append(obj_data['width_pixel'])
            real_dims.append(obj_data['width_m'])
        
        # Add height/length dimension if available
        if obj_data.get('height_pixel', 0) > 0 and obj_data.get('length_m', 0) > 0:
            pixel_dims.append(obj_data['height_pixel'])
            real_dims.append(obj_data['length_m'])
        
        # Add area-derived dimension if available (using sqrt for linear scaling)
        if obj_data.get('area_pixel', 0) > 0 and obj_data.get('area_m', 0) > 0:
            pixel_dims.append(math.sqrt(obj_data['area_pixel']))
            real_dims.append(math.sqrt(obj_data['area_m']))
    
    # If no valid dimensions found, return error
    if not pixel_dims:
        raise ValueError("No valid dimensions found in reference objects")
    
    # Convert to numpy arrays for matrix operations
    pixel_dims = np.array(pixel_dims)
    real_dims = np.array(real_dims)
    
    # Least squares solution (no intercept):
    # We want to find scale_factor such that real_dims ≈ scale_factor * pixel_dims
    # This is equivalent to minimizing sum((real_dims - scale_factor * pixel_dims)²)
    
    # The analytical solution is:
    # scale_factor = (pixel_dims·real_dims) / (pixel_dims·pixel_dims)
    scale_factor = np.dot(pixel_dims, real_dims) / np.dot(pixel_dims, pixel_dims)
    
    return float(scale_factor)


def calculate_scale_by_median_ratio(reference_objects: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculate scale factor using the median of individual object ratios.
    
    This method is robust against outliers as it takes the median rather than mean
    of all individual scale factors computed from each object.
    
    Args:
        reference_objects: Dictionary of reference objects with their measurements
        
    Returns:
        Median scale factor in meters/pixel
    """
    if not reference_objects:
        raise ValueError("No reference objects provided")
    
    # Calculate individual scale factors
    scale_factors = []
    
    for obj_name, obj_data in reference_objects.items():
        obj_scales = []
        
        # Calculate scale from width if available
        if obj_data.get('width_pixel', 0) > 0 and obj_data.get('width_m', 0) > 0:
            obj_scales.append(obj_data['width_m'] / obj_data['width_pixel'])
        
        # Calculate scale from height/length if available
        if obj_data.get('height_pixel', 0) > 0 and obj_data.get('length_m', 0) > 0:
            obj_scales.append(obj_data['length_m'] / obj_data['height_pixel'])
        
        # Calculate scale from area if available
        if obj_data.get('area_pixel', 0) > 0 and obj_data.get('area_m', 0) > 0:
            obj_scales.append(math.sqrt(obj_data['area_m'] / obj_data['area_pixel']))
        
        # Add the average scale for this object if we have measurements
        if obj_scales:
            scale_factors.append(sum(obj_scales) / len(obj_scales))
    
    # If no valid scale factors, return error
    if not scale_factors:
        raise ValueError("No valid scale factors could be calculated")
    
    # Return the median scale factor
    return float(np.median(scale_factors))
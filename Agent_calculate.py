"""
Agent_Calculation.py - Agent for real-world spatial metric calculation in remote sensing imagery.

This module implements an agent that selects an appropriate algorithm for scale factor calculation
and applies it to determine real-world dimensions from pixel measurements.
"""

import os
import json
from typing import Dict, Any, Optional, List
import numpy as np
from openai import OpenAI

# Import the API algorithms
import API_Algorithm as alg

class SpatialMetricCalculationAgent:
    """
    Agent for calculating real-world spatial metrics from remote sensing imagery.
    
    This agent dynamically selects the most appropriate scale calculation algorithm 
    based on the characteristics of the reference objects and the nature of the query.
    
    It can either use the three pre-defined algorithms from API_Algorithm or develop
    a custom algorithm when necessary.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the Spatial Metric Calculation Agent.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4 calls (optional, will use env var if not provided)
        """
        # Initialize OpenAI client for advanced reasoning and algorithm selection
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("Warning: No OpenAI API key provided. Advanced reasoning capabilities will be limited.")
    
    def _convert_numpy_types(self, obj):
        """
        Convert NumPy types to standard Python types for JSON serialization.
        
        Args:
            obj: Object containing potential NumPy data types
            
        Returns:
            Object with NumPy types converted to standard Python types
        """
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self._convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    def select_algorithm(self, reference_objects: Dict[str, Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Select the most appropriate algorithm for scale calculation based on the reference objects and query.
        
        This method analyzes the characteristics of the reference objects and uses GPT-4 (if available)
        to make an informed decision on which algorithm to use.
        
        Args:
            reference_objects: Dictionary of reference objects with their measurements
            query: The original user query
            
        Returns:
            Dictionary with selected algorithm and reasoning
        """
        if not reference_objects:
            raise ValueError("No reference objects provided")
        
        # Analyze the characteristics of reference objects
        # Count objects and categorize them
        object_types = {}
        large_objects = 0
        small_objects = 0
        
        for obj_name, obj_data in reference_objects.items():
            # Get base object type
            obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
            
            # Count by type
            if obj_type not in object_types:
                object_types[obj_type] = 0
            object_types[obj_type] += 1
            
            # Categorize by size
            area_pixel = obj_data.get('area_pixel', 0)
            if area_pixel > 5000:
                large_objects += 1
            else:
                small_objects += 1
        
        # If GPT-4 is available, use it for more nuanced decision-making
        if self.openai_client:
            try:
                # Prepare reference objects for API (remove large data like masks)
                api_reference_objects = {}
                for obj_name, obj_data in reference_objects.items():
                    api_reference_objects[obj_name] = {
                        k: v for k, v in obj_data.items() 
                        if k not in ['mask'] and not isinstance(v, np.ndarray)
                    }
                
                api_reference_objects = self._convert_numpy_types(api_reference_objects)
                
                # Create prompt for GPT-4
                prompt = f"""
                You are a spatial analysis expert tasked with selecting the optimal algorithm for calculating scale factor (meters/pixel) in remote sensing imagery.
                
                USER QUERY: "{query}"
                
                REFERENCE OBJECTS AVAILABLE:
                {json.dumps(api_reference_objects, indent=2)}
                
                OBJECT ANALYSIS:
                - Total reference objects: {len(reference_objects)}
                - Object types: {json.dumps(object_types)}
                - Large objects (>5000px area): {large_objects}
                - Small objects: {small_objects}
                
                Available algorithms:
                
                1. WEIGHTED AREA (calculate_scale_by_weighted_area):
                   - Weights each reference object's scale factor by its pixel area
                   - Formula: S = ∑(Wi * Si) where Wi = Ai / ∑Aj and Si = sqrt(area_m / area_pixel)
                   - Advantages: Gives more influence to larger objects which typically have more reliable measurements
                   - Best for: Scenes with objects of varying sizes where larger objects should be trusted more
                
                2. LEAST SQUARES (calculate_scale_by_least_squares):
                   - Uses least squares optimization to find the scale factor that minimizes error across all measurements
                   - Considers all linear dimensions (width, height, sqrt(area))
                   - Advantages: Mathematically optimal solution that minimizes overall error
                   - Best for: Scenes with consistent measurement quality across objects
                
                3. MEDIAN RATIO (calculate_scale_by_median_ratio):
                   - Takes the median of all individual object scale factors
                   - Advantages: Robust against outliers and extreme values
                   - Best for: Scenes with potential outliers or inconsistent measurements
                
                4. CUSTOM ALGORITHM:
                   - If none of the above are optimal, you can propose a custom algorithm
                   - Describe the algorithm clearly and explain why it's better than the pre-defined options
                
                Select the most appropriate algorithm and explain your reasoning in detail.
                Focus on:
                1. Why this algorithm is best for these specific reference objects
                2. Any potential concerns with the selected approach
                3. What makes this algorithm better than the alternatives for this case
                
                If choosing a custom algorithm, describe it thoroughly.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a specialized spatial analysis system that selects optimal algorithms for scale calculation in remote sensing."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                gpt4_analysis = response.choices[0].message.content
                
                # Extract the algorithm recommendation from GPT-4's response
                algorithm_name = None
                reasoning = gpt4_analysis
                
                # Check for explicit algorithm mentions
                if "weighted area" in gpt4_analysis.lower() and "recommend" in gpt4_analysis.lower():
                    algorithm_name = "weighted_area"
                elif "least squares" in gpt4_analysis.lower() and "recommend" in gpt4_analysis.lower():
                    algorithm_name = "least_squares"
                elif "median ratio" in gpt4_analysis.lower() and "recommend" in gpt4_analysis.lower():
                    algorithm_name = "median_ratio"
                elif "custom algorithm" in gpt4_analysis.lower() and "recommend" in gpt4_analysis.lower():
                    algorithm_name = "custom"
                
                # If no algorithm was clearly identified, perform basic selection
                if not algorithm_name:
                    # Default logic based on object characteristics
                    if len(object_types) > 2:
                        # With diverse object types, median is often safer
                        algorithm_name = "median_ratio"
                    elif large_objects >= 3:
                        # With several large objects, weighted area works well
                        algorithm_name = "weighted_area"
                    else:
                        # Default to least squares for balanced cases
                        algorithm_name = "least_squares"
                
                return {
                    "selected_algorithm": algorithm_name,
                    "reasoning": reasoning,
                    "object_analysis": {
                        "total_objects": len(reference_objects),
                        "object_types": object_types,
                        "large_objects": large_objects,
                        "small_objects": small_objects
                    }
                }
                
            except Exception as e:
                print(f"Error in GPT-4 algorithm selection: {e}")
                # Fall back to basic selection logic
        
        # Basic selection logic (used when GPT-4 is unavailable or fails)
        if len(object_types) > 2:
            # With diverse object types, median is often safer
            algorithm_name = "median_ratio"
            reasoning = "Selected median ratio algorithm because multiple object types are present, which makes it more robust against variations in scale factors between different object categories."
        elif large_objects >= 3:
            # With several large objects, weighted area works well
            algorithm_name = "weighted_area"
            reasoning = "Selected weighted area algorithm because multiple large objects are present. Larger objects typically provide more reliable measurements due to higher pixel counts."
        else:
            # Default to least squares for balanced cases
            algorithm_name = "least_squares"
            reasoning = "Selected least squares algorithm as a balanced approach since there's no strong indication for either weighted area or median methods."
        
        return {
            "selected_algorithm": algorithm_name,
            "reasoning": reasoning,
            "object_analysis": {
                "total_objects": len(reference_objects),
                "object_types": object_types,
                "large_objects": large_objects,
                "small_objects": small_objects
            }
        }
    
    def calculate_scale_factor(self, reference_objects: Dict[str, Dict[str, Any]], query: str = "") -> Dict[str, Any]:
        """
        Select an algorithm and calculate the scale factor for the given reference objects.
        
        Args:
            reference_objects: Dictionary of reference objects with their measurements
            query: The original user query (used for algorithm selection)
            
        Returns:
            Dictionary with scale factor and calculation details
        """
        if not reference_objects:
            raise ValueError("No reference objects provided")
        
        # Select the most appropriate algorithm
        algorithm_selection = self.select_algorithm(reference_objects, query)
        selected_algorithm = algorithm_selection["selected_algorithm"]
        
        # Calculate scale factor using the selected algorithm
        try:
            if selected_algorithm == "weighted_area":
                scale_factor = alg.calculate_scale_by_weighted_area(reference_objects)
                algorithm_name = "Weighted Area"
            elif selected_algorithm == "least_squares":
                scale_factor = alg.calculate_scale_by_least_squares(reference_objects)
                algorithm_name = "Least Squares Optimization"
            elif selected_algorithm == "median_ratio":
                scale_factor = alg.calculate_scale_by_median_ratio(reference_objects)
                algorithm_name = "Median of Individual Ratios"
            elif selected_algorithm == "custom":
                # For custom algorithm, implement the logic described by GPT-4
                # This is just a placeholder - in a real implementation, you would parse
                # the custom algorithm description and implement it
                print("Custom algorithm recommended. Falling back to weighted area.")
                scale_factor = alg.calculate_scale_by_weighted_area(reference_objects)
                algorithm_name = "Custom (fallback to Weighted Area)"
            else:
                raise ValueError(f"Unknown algorithm selected: {selected_algorithm}")
        
        except Exception as e:
            print(f"Error calculating scale with {selected_algorithm} algorithm: {e}")
            print("Trying alternative algorithms...")
            
            # Try all algorithms and use the one that works
            scale_factor = None
            algorithm_name = "Fallback"
            
            try:
                scale_factor = alg.calculate_scale_by_weighted_area(reference_objects)
                algorithm_name = "Weighted Area (fallback)"
            except Exception:
                try:
                    scale_factor = alg.calculate_scale_by_least_squares(reference_objects)
                    algorithm_name = "Least Squares (fallback)"
                except Exception:
                    try:
                        scale_factor = alg.calculate_scale_by_median_ratio(reference_objects)
                        algorithm_name = "Median Ratio (fallback)"
                    except Exception:
                        raise ValueError("All scale calculation algorithms failed")
        
        # Calculate individual object scale factors for comparison
        individual_scales = {}
        for obj_name, obj_data in reference_objects.items():
            obj_scales = {}
            
            # Area-based scale
            if obj_data.get('area_pixel', 0) > 0 and obj_data.get('area_m', 0) > 0:
                obj_scales['area'] = np.sqrt(obj_data['area_m'] / obj_data['area_pixel'])
            
            # Width-based scale
            if obj_data.get('width_pixel', 0) > 0 and obj_data.get('width_m', 0) > 0:
                obj_scales['width'] = obj_data['width_m'] / obj_data['width_pixel']
            
            # Length-based scale
            if obj_data.get('height_pixel', 0) > 0 and obj_data.get('length_m', 0) > 0:
                obj_scales['length'] = obj_data['length_m'] / obj_data['height_pixel']
            
            if obj_scales:
                individual_scales[obj_name] = obj_scales
        
        # Return the result with comprehensive details
        return {
            "scale_factor": scale_factor,
            "units": "meters/pixel",
            "algorithm": selected_algorithm,
            "algorithm_name": algorithm_name,
            "reasoning": algorithm_selection["reasoning"],
            "individual_scales": individual_scales,
            "object_analysis": algorithm_selection["object_analysis"]
        }
    
    def calculate_real_world_dimensions(self, pixel_dimensions: Dict[str, float], scale_factor: float) -> Dict[str, float]:
        """
        Calculate real-world dimensions from pixel dimensions using the scale factor.
        
        Args:
            pixel_dimensions: Dictionary of pixel dimensions (width, height, area, etc.)
            scale_factor: Scale factor in meters/pixel
            
        Returns:
            Dictionary of real-world dimensions in meters
        """
        real_world = {}
        
        if 'width' in pixel_dimensions:
            real_world['width_m'] = pixel_dimensions['width'] * scale_factor
            
        if 'height' in pixel_dimensions:
            real_world['height_m'] = pixel_dimensions['height'] * scale_factor
            
        if 'perimeter' in pixel_dimensions:
            real_world['perimeter_m'] = pixel_dimensions['perimeter'] * scale_factor
            
        # Area conversion (square the scale factor for area)
        if 'area' in pixel_dimensions:
            real_world['area_m2'] = pixel_dimensions['area'] * (scale_factor ** 2)
            
        # Volume conversion if height is provided
        if 'volume' in pixel_dimensions and 'height_m' in real_world:
            real_world['volume_m3'] = pixel_dimensions['volume'] * (scale_factor ** 3)
            
        return real_world
    
    def run(self, reference_objects: Dict[str, Dict[str, Any]], pixel_dimensions: Dict[str, float], query: str = "") -> Dict[str, Any]:
        """
        Execute the complete workflow: select algorithm, calculate scale factor, and convert dimensions.
        
        Args:
            reference_objects: Dictionary of reference objects with their measurements
            pixel_dimensions: Dictionary of pixel dimensions for the object of interest
            query: The original user query
            
        Returns:
            Dictionary with scale factor, calculation details, and real-world dimensions
        """
        # Step 1: Calculate scale factor
        scale_result = self.calculate_scale_factor(reference_objects, query)
        scale_factor = scale_result["scale_factor"]
        
        # Step 2: Calculate real-world dimensions
        real_world_dimensions = self.calculate_real_world_dimensions(pixel_dimensions, scale_factor)
        
        # Step 3: Prepare comprehensive result
        return {
            "scale_factor": scale_factor,
            "units": "meters/pixel",
            "algorithm": scale_result["algorithm"],
            "algorithm_name": scale_result["algorithm_name"],
            "reasoning": scale_result["reasoning"],
            "pixel_dimensions": pixel_dimensions,
            "real_world_dimensions": real_world_dimensions
        }


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = SpatialMetricCalculationAgent()
    
    # Example reference objects (from ReferenceDetectionAgent)
    reference_objects = {
  "boeing_787_1": {
    "width_pixel": 336.3642883300781,
    "height_pixel": 539.3071899414062,
    "area_pixel": 46958,
    "width_m": 60.1,
    "length_m": 56.7,
    "area_m": 690.88
  },
  "airbus_320_1": {
    "width_pixel": 177.587158203125,
    "height_pixel": 260.5806884765625,
    "area_pixel": 16415,
    "width_m": 35.8,
    "length_m": 37.57,
    "area_m": 282.36
  },
  "boeing_787_2": {
    "width_pixel": 358.55816650390625,
    "height_pixel": 503.2900085449219,
    "area_pixel": 42466,
    "width_m": 60.1,
    "length_m": 56.7,
    "area_m": 690.88
  },
  "airbus_320_2": {
    "width_pixel": 215.10107421875,
    "height_pixel": 225.98960876464844,
    "area_pixel": 13625,
    "width_m": 35.8,
    "length_m": 37.57,
    "area_m": 282.36
  },
  "airbus_320_3": {
    "width_pixel": 218.79861450195312,
    "height_pixel": 248.78758239746094,
    "area_pixel": 20192,
    "width_m": 35.8,
    "length_m": 37.57,
    "area_m": 282.36
  }
}
    
    # Example pixel dimensions of object of interest
    pixel_dimensions = {
        "width": 250,
        "height": 400,
        "area": 100000
    }
    
    # Example query
    query = "What is the real-world area of the building in the image?"
    
    # Run the agent
    result = agent.run(reference_objects, pixel_dimensions, query)
    
    print(f"Scale Factor: {result['scale_factor']:.6f} {result['units']}")
    print(f"Algorithm: {result['algorithm_name']}")
    print(f"Real-world dimensions: {result['real_world_dimensions']}")
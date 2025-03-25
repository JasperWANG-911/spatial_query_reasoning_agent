"""
Agent for Reference Detection (Simplified Version).
This module implements a streamlined agent that detects and selects reference objects.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from openai import OpenAI

# Import the API modules
from API_Grounded_RS_VLM import GeoChat, analyse_caption_for_references, reference_object_list
from API_Vision_Specialist import SAM2
import API_Reference_Object_Selection as ref_selection

class ReferenceDetectionAgent:
    """
    Simplified agent for detecting and selecting reference objects in remote sensing imagery.
    
    Workflow:
    1. Image -> GeoChat captioning -> List of potential objects
    2. GPT-4 analysis of caption -> Matched reference objects
    3. GeoChat bounding box generation -> Objects with coordinates
    4. SAM2 segmentation -> Precise masks and measurements
    5. Dynamic selection of reference objects based on query
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the Reference Detection Agent.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4 calls (optional, will use env var if not provided)
        """
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize API components
        self.geochat = GeoChat()
        self.sam = SAM2()
        
        # Store reference object list
        self.reference_object_list = reference_object_list
    
    def execute_fixed_workflow(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Execute the fixed part of the workflow to detect reference objects.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detected reference objects with comprehensive information
        """
        # Step 1: Get image description from GeoChat
        image_description = self.geochat.generate_captionning(image_path)
        print(f"Image description: {image_description}")
        
        # Step 2: Analyze the description to find potential reference objects
        matched_objects = analyse_caption_for_references(image_description)
        print(f"Matched reference objects: {matched_objects}")
        
        # Step 3: Get bounding boxes for the reference objects
        objects_with_bbox = self.geochat.generate_bbox(image_path, matched_objects)
        print(f"Objects with bounding boxes: {len(objects_with_bbox)}")
        
        # Step 4: Process with SAM2 to get masks and measurements
        reference_objects = []
        
        for obj in objects_with_bbox:
            obj_name = obj['obj_name']
            bbox = obj['bbox']
            
            try:
                # Load image for SAM
                image = self.sam.load_image(image_path)
                self.sam.set_image(image)
                
                # Get mask
                x1, y1, x2, y2 = bbox
                mask = self.sam.predict_mask(x1, y1, x2, y2)
                
                # Get smallest bounding box and measurements
                box, rect = self.sam.get_smallest_bounding_box(mask)
                
                if not rect:
                    continue
                
                # Calculate measurements
                width_pixel = min(rect[1])
                height_pixel = max(rect[1])
                area_pixel = self.sam.compute_mask_pixel(mask)
                
                # Find reference information from predefined list
                base_obj_type = obj_name.split('_')[0]  # Extract base type (e.g., 'car' from 'car_1')
                ref_info = next((ref for ref in self.reference_object_list if ref['type'] == base_obj_type), None)
                
                # Create comprehensive object info
                obj_info = {
                    'obj': obj_name,
                    'mask': mask,
                    'bbox': bbox,
                    'width_pixel': width_pixel,
                    'height_pixel': height_pixel,
                    'area_pixel': area_pixel
                }
                
                # Add reference dimensions if available
                if ref_info:
                    if 'dimensions' in ref_info:
                        dimensions = ref_info['dimensions']
                        if 'width' in dimensions:
                            obj_info['width_m'] = dimensions['width']
                        if 'length' in dimensions:
                            obj_info['length_m'] = dimensions['length']
                    if 'area' in ref_info:
                        obj_info['area_m'] = ref_info['area']
                    if 'reliability' in ref_info:
                        obj_info['reliability'] = ref_info['reliability']
                
                # Add quality score
                obj_info['quality'] = ref_selection.estimate_object_quality(obj_info)
                
                reference_objects.append(obj_info)
                
            except Exception as e:
                print(f"Error processing object {obj_name}: {e}")
                continue
        
        return reference_objects
    
    def dynamically_select_references(
        self, 
        reference_objects: List[Dict[str, Any]], 
        query: str,
        target_object: Optional[str] = None,
        target_point: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Dynamically select appropriate reference objects based on the query and context.
        
        Args:
            reference_objects: List of detected reference objects
            query: The original user query
            target_object: Optional name of the target object of interest
            target_point: Optional point of interest coordinates as (x, y)
            
        Returns:
            Selected list of reference objects
        """
        # First, get GPT-4's decision on how to select and filter references
        selection_strategy = self._get_selection_strategy(reference_objects, query, target_object)
        print(f"Selected strategy: {selection_strategy['strategy_name']}")
        
        # Apply the selected strategy
        strategy = selection_strategy['strategy']
        filtered_objects = reference_objects.copy()
        
        for step in strategy:
            step_type = step['type']
            
            if step_type == 'filter_by_area':
                min_area = step.get('min_area', 0)
                max_area = step.get('max_area', None)
                filtered_objects = ref_selection.filter_by_area(filtered_objects, min_area, max_area)
                print(f"After filtering by area: {len(filtered_objects)} objects")
                
            elif step_type == 'prioritize_by_size':
                prefer_larger = step.get('prefer_larger', True)
                filtered_objects = ref_selection.prioritize_by_size(filtered_objects, prefer_larger)
                # Take top N if specified
                top_n = step.get('top_n', len(filtered_objects))
                filtered_objects = filtered_objects[:top_n]
                print(f"After prioritizing by size: selected {len(filtered_objects)} objects")
                
            elif step_type == 'filter_by_known_dimensions':
                filtered_objects = ref_selection.filter_by_known_dimensions(filtered_objects, self.reference_object_list)
                print(f"After filtering by known dimensions: {len(filtered_objects)} objects")
                
            elif step_type == 'select_diverse_references':
                min_objects = step.get('min_objects', 3)
                filtered_objects = ref_selection.select_diverse_references(
                    filtered_objects, min_objects
                )
                print(f"After selecting diverse objects: {len(filtered_objects)} objects")
                
            elif step_type == 'custom_operation':
                # Execute a custom operation directly from the GPT-4 plan
                operation = step.get('operation', {})
                operation_type = operation.get('type', '')
                print(f"Executing custom operation: {operation_type}")
                
                try:
                    if operation_type == 'filter_by_ratio':
                        # Filter objects based on aspect ratio
                        min_ratio = operation.get('min_ratio', 0)
                        max_ratio = operation.get('max_ratio', float('inf'))
                        filtered_objects = [
                            obj for obj in filtered_objects 
                            if obj.get('width_pixel', 0) > 0 and obj.get('height_pixel', 0) > 0 and
                            min_ratio <= (obj.get('width_pixel', 1) / obj.get('height_pixel', 1)) <= max_ratio
                        ]
                        print(f"After filtering by ratio: {len(filtered_objects)} objects")
                    
                    elif operation_type == 'custom_scoring':
                        # Score objects based on custom criteria
                        scoring_weights = operation.get('weights', {})
                        size_weight = scoring_weights.get('size', 1.0)
                        reliability_weight = scoring_weights.get('reliability', 1.0)
                        quality_weight = scoring_weights.get('quality', 1.0)
                        
                        # Calculate scores
                        scored_objects = []
                        for obj in filtered_objects:
                            score = 0
                            # Size component
                            if size_weight > 0:
                                # Normalize area to 0-1 range (assuming max area is 100,000 pixels)
                                area_score = min(1.0, obj.get('area_pixel', 0) / 100000)
                                score += size_weight * area_score
                            
                            # Reliability component
                            if reliability_weight > 0:
                                score += reliability_weight * obj.get('reliability', 0.5)
                            
                            # Quality component
                            if quality_weight > 0:
                                score += quality_weight * obj.get('quality', 0.5)
                            
                            scored_objects.append((obj, score))
                        
                        # Sort by score and take top N
                        top_n = operation.get('top_n', len(filtered_objects))
                        filtered_objects = [obj for obj, _ in sorted(scored_objects, key=lambda x: x[1], reverse=True)[:top_n]]
                        print(f"After custom scoring: selected top {len(filtered_objects)} objects")
                    
                    elif operation_type == 'combine_similar_objects':
                        # Group by object type
                        object_groups = {}
                        for obj in filtered_objects:
                            obj_type = obj.get('obj', '').split('_')[0]  # Base type (e.g., 'car' from 'car_1')
                            if obj_type not in object_groups:
                                object_groups[obj_type] = []
                            object_groups[obj_type].append(obj)
                        
                        # For each group, select the best representative
                        combined_objects = []
                        for obj_type, objects in object_groups.items():
                            if len(objects) == 1:
                                combined_objects.append(objects[0])
                            else:
                                # Use quality score to find the best representative
                                best_obj = max(objects, key=lambda x: x.get('quality', 0))
                                combined_objects.append(best_obj)
                        
                        filtered_objects = combined_objects
                        print(f"After combining similar objects: {len(filtered_objects)} objects")
                    
                    else:
                        print(f"Unknown custom operation type: {operation_type}")
                
                except Exception as e:
                    print(f"Error executing custom operation: {e}")
        
        return filtered_objects
    
    def _get_selection_strategy(
        self, 
        reference_objects: List[Dict[str, Any]], 
        query: str,
        target_object: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use GPT-4 to dynamically decide how to select and filter reference objects.
        
        Args:
            reference_objects: List of detected reference objects
            query: The original user query
            target_object: Optional name of the target object of interest
            
        Returns:
            Strategy for selecting and filtering reference objects
        """
        # Extract basic information to avoid sending too much data to API
        obj_summaries = []
        for obj in reference_objects:
            # Create a summary without mask data
            summary = {
                'obj': obj.get('obj', ''),
                'width_pixel': obj.get('width_pixel', 0),
                'height_pixel': obj.get('height_pixel', 0),
                'area_pixel': obj.get('area_pixel', 0),
                'width_m': obj.get('width_m', None),
                'length_m': obj.get('length_m', None),
                'area_m': obj.get('area_m', None),
                'reliability': obj.get('reliability', None),
                'quality': obj.get('quality', None)
            }
            obj_summaries.append(summary)
        
        # Create the prompt for GPT-4
        api_descriptions = """
        Available functions in API_Reference_Object_Selection:
        
        1. filter_by_area(reference_objects, min_area=0, max_area=None):
           Filters objects based on their area in pixels.
           
        2. prioritize_by_size(reference_objects, prefer_larger=True): 
           Sorts objects by size, preferring larger or smaller objects.
           
        3. filter_by_known_dimensions(reference_objects, reference_list):
           Filters to only include objects with known real-world dimensions.
           
        4. select_diverse_references(reference_objects, min_objects=3):
           Selects a diverse set of objects, preferring different types.
           
        5. estimate_object_quality(obj):
           Estimates the quality of a reference object (already calculated for each object).
        
        CUSTOM OPERATIONS:
        You can also define custom operations for situations where the predefined functions 
        aren't sufficient.
        
        To use a custom operation, include a step with:
        {
            "type": "custom_operation",
            "reason": "Why this operation is needed",
            "operation": {
                "type": "operation_type",
                // Additional parameters specific to the operation
            }
        }
        
        Available custom operation types:
        
        1. filter_by_ratio - Filter objects based on their aspect ratio:
           {
               "type": "filter_by_ratio",
               "min_ratio": 0.5,
               "max_ratio": 2.0
           }
           
        2. custom_scoring - Score and rank objects based on weighted criteria:
           {
               "type": "custom_scoring",
               "weights": {
                   "size": 1.0,
                   "reliability": 1.5,
                   "quality": 2.0
               },
               "top_n": 3
           }
           
        3. combine_similar_objects - Combine similar objects into representatives:
           {
               "type": "combine_similar_objects"
           }
        """
        
        prompt = f"""
        You are a dynamic agent for reference object selection in remote sensing imagery.
        
        Original query: {query}
        
        You've detected the following reference objects:
        {json.dumps(obj_summaries, indent=2)}
        
        {api_descriptions}
        
        Based on the query and detected objects, create a strategy for selecting the most appropriate
        reference objects. Return your strategy as a JSON object with the following structure:
        
        {{
            "strategy_name": "Brief name of your strategy",
            "reasoning": "Your reasoning about why you chose this strategy",
            "strategy": [
                {{
                    "type": "function_name",  // One of the available functions or "custom_operation"
                    "reason": "Why this step is needed",
                    // Additional parameters specific to the function:
                    "min_area": 1000,  // For filter_by_area
                    "max_area": 50000,  // For filter_by_area
                    "prefer_larger": true,  // For prioritize_by_size
                    "top_n": 5,  // Number of top objects to keep after prioritizing
                    "min_objects": 3,  // For select_diverse_references
                    
                    // For custom_operation:
                    "operation": {{
                        "type": "custom_scoring",
                        "weights": {{"size": 1.0, "reliability": 1.5, "quality": 2.0}},
                        "top_n": 3
                    }}
                }}
            ]
        }}
        
        Choose functions, custom operations, and parameters that best suit the specific query and available objects.
        Create a strategy of 2-4 steps that will effectively filter and select the most appropriate reference objects.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a specialized assistant for spatial analysis and reference object detection in remote sensing."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response as JSON
        try:
            result = json.loads(response.choices[0].message.content)
            print(f"Selection strategy: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"Error parsing GPT-4 response: {e}")
            # Return a default strategy if parsing fails
            return {
                "strategy_name": "Default quality-based strategy",
                "reasoning": "Simple strategy focusing on quality and diversity",
                "strategy": [
                    {
                        "type": "filter_by_known_dimensions",
                        "reason": "Only use objects with known real-world dimensions"
                    },
                    {
                        "type": "custom_operation",
                        "reason": "Rank objects by quality score",
                        "operation": {
                            "type": "custom_scoring",
                            "weights": {"quality": 1.0, "reliability": 1.0},
                            "top_n": 5
                        }
                    },
                    {
                        "type": "select_diverse_references",
                        "reason": "Ensure diversity in the selected references",
                        "min_objects": 3
                    }
                ]
            }
    
    def run(self, image_path: str, query: str, target_object: Optional[str] = None, target_point: Optional[Tuple[int, int]] = None) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Run the complete reference detection workflow.
        
        Args:
            image_path: Path to the input image
            query: The user query
            target_object: Optional name of the target object of interest
            target_point: Optional point of interest coordinates as (x, y)
            
        Returns:
            Dictionary of selected reference objects with their measurements
        """
        # Step 1: Execute fixed workflow to detect reference objects
        print("Starting fixed workflow to detect reference objects...")
        reference_objects = self.execute_fixed_workflow(image_path)
        print(f"Fixed workflow complete, detected {len(reference_objects)} reference objects")
        
        # Early return if no objects detected
        if not reference_objects:
            print("No reference objects detected. Aborting.")
            return {}
        
        # Step 2: Dynamically select appropriate reference objects
        print("Starting dynamic selection of reference objects...")
        selected_objects = self.dynamically_select_references(
            reference_objects, query, target_object, target_point
        )
        print(f"Dynamic selection complete, selected {len(selected_objects)} reference objects")
        
        # Step 3: Format the final output
        final_output = {}
        
        for obj in selected_objects:
            obj_name = obj['obj']
            
            # Create a dictionary with measurements
            measurements = {
                'width_pixel': obj.get('width_pixel', 0),
                'height_pixel': obj.get('height_pixel', 0),
                'area_pixel': obj.get('area_pixel', 0)
            }
            
            # Add real-world dimensions if available
            if 'width_m' in obj:
                measurements['width_m'] = obj['width_m']
            if 'length_m' in obj:
                measurements['length_m'] = obj['length_m']
            if 'area_m' in obj:
                measurements['area_m'] = obj['area_m']
            
            final_output[obj_name] = measurements
        
        return final_output


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = ReferenceDetectionAgent()
    
    # Run the complete workflow
    image_path = "example_image.jpg"
    query = "What is building's real-world footprint"
    
    result = agent.run(image_path, query)
    print(f"Final result: {json.dumps(result, indent=2)}")
"""
Agent for Reference Detection.
This module implements an agent that detects and selects reference objects in remote sensing imagery.
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
    Agent for detecting and selecting reference objects in remote sensing imagery.
    
    The agent uses a hybrid approach:
    1. A fixed workflow to detect reference objects (GeoChat + SAM2)
    2. A dynamic approach to select the most appropriate references for the query
    
    Workflow:
    1. Image -> GeoChat captioning -> List of potential objects
    2. GPT-4 analysis of caption -> Matched reference objects
    3. GeoChat bounding box generation -> Objects with coordinates
    4. SAM2 segmentation -> Precise masks and measurements
    5. Dynamic selection of reference objects based on query
    """
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
        self.geochat = GeoChat(api_key=self.openai_api_key)
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
                if '_' in obj_name:
                    # 找到最后一个下划线的位置
                    last_underscore_pos = obj_name.rfind('_')
                    # 提取最后一个下划线之前的所有内容
                    base_obj_type = obj_name[:last_underscore_pos]
                else:
                    # 如果没有下划线，使用整个名称
                    base_obj_type = obj_name
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
                
                # 确保所有对象都有真实世界尺寸信息
                if ref_info:
                    # 添加尺寸信息
                    if 'dimensions' in ref_info:
                        dimensions = ref_info['dimensions']
                        if 'width' in dimensions:
                            obj_info['width_m'] = dimensions['width']
                        if 'length' in dimensions:
                            obj_info['length_m'] = dimensions['length']
                    else:
                        # 如果没有dimensions字段，添加默认值
                        obj_info['width_m'] = 1.0  # 默认值
                        obj_info['length_m'] = 2.0  # 默认值
                        print(f"Warning: No dimension information for {base_obj_type}, using defaults")
                    
                    # 添加面积信息
                    if 'area' in ref_info:
                        obj_info['area_m'] = ref_info['area']
                    elif 'width_m' in obj_info and 'length_m' in obj_info:
                        # 如果没有area但有宽度和长度，计算面积
                        obj_info['area_m'] = obj_info['width_m'] * obj_info['length_m']
                    else:
                        # 如果没有足够信息，添加默认值
                        obj_info['area_m'] = 2.0  # 默认值
                        print(f"Warning: No area information for {base_obj_type}, using default")
                    
                    # 添加可靠性信息
                    if 'reliability' in ref_info:
                        obj_info['reliability'] = ref_info['reliability']
                    else:
                        obj_info['reliability'] = 0.5  # 默认可靠性
                else:
                    # 如果在预定义列表中找不到该对象类型，使用默认值
                    print(f"Warning: {base_obj_type} not found in reference object list, using default values")
                    obj_info['width_m'] = 1.0
                    obj_info['length_m'] = 2.0
                    obj_info['area_m'] = 2.0
                    obj_info['reliability'] = 0.3
                
                # 计算纵横比用于质量评估
                obj_info['aspect_ratio'] = height_pixel / width_pixel if width_pixel > 0 else 1.0
                
                # 添加质量评分
                obj_info['quality'] = ref_selection.estimate_object_quality(obj_info)
                
                reference_objects.append(obj_info)
                
            except Exception as e:
                print(f"Error processing object {obj_name}: {e}")
                continue
        
        if reference_objects:
            try:
                # visualise all detected reference objects
                visualization_path = os.path.join(os.path.dirname(image_path), "all_detected_objects.jpg")
                self.sam.visualize_multiple_objects(image_path, reference_objects, save_path=visualization_path)
                print(f"All detected reference objects visualization saved to {visualization_path}")
            except Exception as e:
                print(f"Warning: Could not visualize reference objects: {e}")

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
        
        # Debug log all objects at the start
        print(f"Starting with {len(filtered_objects)} objects")
        for i, obj in enumerate(filtered_objects):
            obj_type = obj.get('obj', '').split('_')[0]
            print(f"  {i+1}. {obj.get('obj', 'Unknown')} - Type: {obj_type}, Area: {obj.get('area_pixel', 0)} px, Quality: {obj.get('quality', 0):.2f}")
        
        for step in strategy:
            step_type = step['type']
            
            # Handle predefined API functions
            if step_type in ['check_scale_consistency', 'select_diverse_objects', 'balance_object_types', 'prioritize_larger_objects']:
                try:
                    # Extract parameters for the function call, excluding non-parameter fields
                    params = {k: v for k, v in step.items() if k not in ['type', 'reason']}
                    
                    # 限制prioritize_larger_objects的min_area_pixel参数
                    if step_type == 'prioritize_larger_objects' and 'min_area_pixel' in params:
                        max_allowed = 300  # 设置允许的最大阈值
                        if params['min_area_pixel'] > max_allowed:
                            original_value = params['min_area_pixel']
                            params['min_area_pixel'] = max_allowed
                            print(f"Limiting min_area_pixel from {original_value} to {max_allowed}")
                    
                    # Call the appropriate function from the API
                    func = getattr(ref_selection, step_type)
                    filtered_objects = func(filtered_objects, **params)
                    
                    print(f"After {step_type}: {len(filtered_objects)} objects")
                except Exception as e:
                    print(f"Error executing {step_type}: {e}")
            
            # Handle custom operations
            elif step_type == 'custom_operation':
                operation = step.get('operation', {})
                operation_type = operation.get('type', '')
                
                print(f"Executing custom operation: {operation_type}")
                
                try:
                    if operation_type == 'filter_by_area':
                        min_area = operation.get('min_area', 0)
                        max_area = operation.get('max_area', float('inf'))
                        
                        # 限制min_area参数
                        max_allowed_min_area = 1000
                        if min_area > max_allowed_min_area:
                            print(f"Limiting min_area from {min_area} to {max_allowed_min_area}")
                            min_area = max_allowed_min_area
                        
                        filtered_objects = [
                            obj for obj in filtered_objects 
                            if min_area <= obj.get('area_pixel', 0) <= max_area
                        ]
                        print(f"After filtering by area: {len(filtered_objects)} objects")
                    
                    elif operation_type == 'filter_by_aspect_ratio':
                        min_ratio = operation.get('min_ratio', 0)
                        max_ratio = operation.get('max_ratio', float('inf'))
                        filtered_objects = [
                            obj for obj in filtered_objects 
                            if obj.get('width_pixel', 0) > 0 and obj.get('height_pixel', 0) > 0 and
                            min_ratio <= (obj.get('width_pixel', 1) / obj.get('height_pixel', 1)) <= max_ratio
                        ]
                        print(f"After filtering by aspect ratio: {len(filtered_objects)} objects")
                    
                    elif operation_type == 'prioritize_by_quality':
                        filtered_objects = sorted(
                            filtered_objects, 
                            key=lambda x: x.get('quality', 0), 
                            reverse=True
                        )
                        # Take top N if specified
                        top_n = operation.get('top_n', len(filtered_objects))
                        filtered_objects = filtered_objects[:min(top_n, len(filtered_objects))]
                        print(f"After prioritizing by quality: {len(filtered_objects)} objects")
                    
                    elif operation_type == 'filter_by_known_dimensions':
                        filtered_objects = [
                            obj for obj in filtered_objects
                            if ('width_m' in obj or 'length_m' in obj or 'area_m' in obj)
                        ]
                        print(f"After filtering by known dimensions: {len(filtered_objects)} objects")

                    elif operation_type == 'custom_scoring':
                        weights = operation.get('weights', {})
                        size_weight = weights.get('size', 1.0)
                        reliability_weight = weights.get('reliability', 1.0)
                        quality_weight = weights.get('quality', 1.0)
                        area_m_weight = weights.get('area_m', 0.0)
                        
                        scored_objects = []
                        for obj in filtered_objects:
                            score = 0
                            
                            # Size component
                            if size_weight > 0:
                                area_score = min(1.0, obj.get('area_pixel', 0) / 100000)
                                score += size_weight * area_score
                            
                            # Reliability component
                            if reliability_weight > 0:
                                score += reliability_weight * obj.get('reliability', 0.5)
                            
                            # Quality component
                            if quality_weight > 0:
                                score += quality_weight * obj.get('quality', 0.5)
                                
                            # Real world area component
                            if area_m_weight > 0 and 'area_m' in obj:
                                area_m_score = min(1.0, obj.get('area_m', 0) / 10000)
                                score += area_m_weight * area_m_score
                            
                            scored_objects.append((obj, score))
                        
                        # Sort by score and take top N
                        top_n = operation.get('top_n', len(filtered_objects))
                        filtered_objects = [obj for obj, _ in sorted(scored_objects, key=lambda x: x[1], reverse=True)[:min(top_n, len(scored_objects))]]
                        print(f"After custom scoring: {len(filtered_objects)} objects")
                    
                    elif operation_type == 'combine_similar_objects':
                        # Group by object type
                        object_groups = {}
                        for obj in filtered_objects:
                            obj_type = obj.get('obj', '').split('_')[0]
                            if obj_type not in object_groups:
                                object_groups[obj_type] = []
                            object_groups[obj_type].append(obj)
                        
                        # For each group, select the best representative
                        combined_objects = []
                        for obj_type, objects in object_groups.items():
                            if len(objects) == 1:
                                combined_objects.append(objects[0])
                            else:
                                best_obj = max(objects, key=lambda x: x.get('quality', 0))
                                combined_objects.append(best_obj)
                        
                        filtered_objects = combined_objects
                        print(f"After combining similar objects: {len(filtered_objects)} objects")
                    
                    elif operation_type == 'prioritize_mix_object_types':
                        # Group by object type
                        object_groups = {}
                        for obj in filtered_objects:
                            obj_type = obj.get('obj', '').split('_')[0]
                            if obj_type not in object_groups:
                                object_groups[obj_type] = []
                            object_groups[obj_type].append(obj)
                        
                        max_per_type = operation.get('max_per_type', 2)
                        max_total = operation.get('max_total', 5)
                        
                        # Select top objects from each type
                        mixed_objects = []
                        for obj_type, objects in object_groups.items():
                            # Sort by quality
                            sorted_objs = sorted(objects, key=lambda x: x.get('quality', 0), reverse=True)
                            # Add top N from each type
                            mixed_objects.extend(sorted_objs[:min(max_per_type, len(sorted_objs))])
                        
                        # If we exceeded max_total, keep only the highest quality ones
                        if len(mixed_objects) > max_total:
                            mixed_objects = sorted(mixed_objects, key=lambda x: x.get('quality', 0), reverse=True)[:max_total]
                        
                        filtered_objects = mixed_objects
                        print(f"After mixing object types: {len(filtered_objects)} objects")
                        
                    elif operation_type.startswith('custom_'):
                        # Execute custom code generated by the GPT-4 model
                        custom_filter_code = operation.get('code', '')
                        if custom_filter_code:
                            try:
                                # Execute the custom filter code
                                custom_filter = self._create_custom_filter(custom_filter_code)
                                if custom_filter:
                                    filtered_objects = custom_filter(filtered_objects, operation)
                                    print(f"After custom filter '{operation_type}': {len(filtered_objects)} objects")
                            except Exception as e:
                                print(f"Error in custom filter: {e}")
                    else:
                        print(f"Unknown custom operation type: {operation_type}")
                
                except Exception as e:
                    print(f"Error executing custom operation: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Debug log remaining objects at the end
        print(f"Final selection: {len(filtered_objects)} objects")
        for i, obj in enumerate(filtered_objects):
            obj_type = obj.get('obj', '').split('_')[0]
            print(f"  {i+1}. {obj.get('obj', 'Unknown')} - Type: {obj_type}, Area: {obj.get('area_pixel', 0)} px, Quality: {obj.get('quality', 0):.2f}")
        
        return filtered_objects

    def _create_custom_filter(self, code_string: str):
        """
        Create a custom filter function from code string provided by GPT-4.
        
        Args:
            code_string: Python code string defining a filter function
            
        Returns:
            Callable filter function or None if creation failed
        """
        try:
            # Define a safe namespace for the function
            namespace = {
                'np': np,
                'List': List,
                'Dict': Dict,
                'Any': Any,
                'Optional': Optional
            }
            
            # Execute the code string in the namespace
            exec(code_string, namespace)
            
            # Return the filter function if it exists
            if 'custom_filter' in namespace and callable(namespace['custom_filter']):
                return namespace['custom_filter']
            else:
                print("Error: custom_filter function not found in generated code")
                return None
        except Exception as e:
            print(f"Error creating custom filter: {e}")
            return None
    
    def _get_selection_strategy(
        self, 
        reference_objects: List[Dict[str, Any]], 
        query: str,
        target_object: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use GPT-4o to dynamically decide how to select and filter reference objects.
        
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
        
        # Get the unique object types
        object_types = set()
        for obj in obj_summaries:
            obj_type = obj.get('obj', '').split('_')[0]  # Extract base type
            object_types.add(obj_type)
        
        # Convert obj_summaries to JSON-serializable format
        obj_summaries_converted = self._convert_numpy_types(obj_summaries)
        
        # Create the prompt for GPT-4
        api_descriptions = """
        You have access to a set of CORE API FUNCTIONS that help with reference object selection:

        1. check_scale_consistency(reference_objects, tolerance=0.3):
           Checks if scale ratios (meters/pixels) among objects of the same type are consistent,
           filtering out objects with inconsistent scales.

        2. select_diverse_objects(reference_objects, min_objects=3, keep_best_per_type=True):
           Selects a diverse set of reference objects, prioritizing different types while
           keeping high-quality objects.

        3. balance_object_types(reference_objects, min_per_type=1, max_total=5):
           Ensures a balance of different object types in the selected references.

        4. prioritize_larger_objects(reference_objects, min_area_pixel=500, min_objects=2):
           Prioritizes larger objects that typically provide more accurate scale references.

        IN ADDITION, you can create CUSTOM OPERATIONS when the core functions are insufficient.
        Create a custom operation using this format:

        {
            "type": "custom_operation",
            "reason": "Why this operation is needed",
            "operation": {
                "type": "operation_type",
                // Additional parameters specific to the operation
            }
        }

        Some useful custom operations include:

        1. filter_by_area:
           {
               "type": "filter_by_area",
               "min_area": 1000,
               "max_area": 50000
           }

        2. filter_by_aspect_ratio:
           {
               "type": "filter_by_aspect_ratio",
               "min_ratio": 0.5,
               "max_ratio": 2.0
           }

        3. prioritize_by_quality:
           {
               "type": "prioritize_by_quality",
               "top_n": 6
           }

        4. custom_scoring:
           {
               "type": "custom_scoring",
               "weights": {
                   "size": 1.0,
                   "reliability": 1.5,
                   "quality": 2.0,
                   "area_m": 1.0
               },
               "top_n": 3
           }

        5. combine_similar_objects:
           {
               "type": "combine_similar_objects"
           }

        6. filter_by_known_dimensions:
           {
               "type": "filter_by_known_dimensions"
           }

        7. prioritize_mix_object_types:
           {
               "type": "prioritize_mix_object_types",
               "max_per_type": 2,
               "max_total": 5
           }

        8. For completely custom behavior, you can define a custom filter:
           {
               "type": "custom_specialized_filter",
               "description": "A custom filter that...",
               "code": "def custom_filter(objects, params):\\n    # Your filtering logic here\\n    return filtered_objects"
           }

        You should choose a mix of CORE API FUNCTIONS and CUSTOM OPERATIONS that best address
        the specific query being analyzed.
        """
        
        prompt = f"""
        You are a dynamic agent for reference object selection in remote sensing imagery.
        
        Original query: "{query}"
        
        You've detected the following reference objects:
        {json.dumps(obj_summaries_converted, indent=2)}
        
        {api_descriptions}
        
        Your task is to create a strategy for selecting the most appropriate reference objects.
        Return your strategy as a JSON object with this structure:
        
        {{
            "strategy_name": "Brief name of your strategy",
            "reasoning": "Detailed reasoning about why you chose this strategy",
            "strategy": [
                {{
                    "type": "function_name_or_custom_operation",
                    "reason": "Why this step is needed",
                    // Additional parameters specific to the function
                }}
            ]
        }}
        
        IMPORTANT STRATEGIC GUIDELINES:
        
        1. For queries about scale calculation:
           - For scale calculation specifically, consider selecting 5-6 diverse reference objects 
             instead of just 2-3 to ensure more robust results.
           - When multiple object types are available (like cars AND tennis courts), prefer larger objects
             like tennis courts over smaller ones like cars, as they typically provide more accurate scale references.
           - Tennis courts (23.77 x 10.97m) provide more stable references than cars (4.5 x 1.8m).
           - Always include at least one object of each detected type for better diversity and reliability.
           
        2. For queries about specific objects:
           - Balance reliability and relevance to the query.
           - When possible, include objects from different parts of the image for better spatial coverage.
           
        3. General considerations:
           - Consider checking scale consistency with check_scale_consistency().
           - Ensure diversity with select_diverse_objects() or custom operations.
           - Larger objects typically have more accurate measurements due to higher pixel counts.
           - If you need very specific filtering beyond the standard operations, create a custom filter.
           
        Be CREATIVE and ADAPTIVE. Design a strategy with 2-4 steps that is SPECIFICALLY TAILORED 
        to the query: "{query}" and the detected objects.
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
                "strategy_name": "Default diverse reference selection",
                "reasoning": "Providing a balanced set of reference objects for general purposes",
                "strategy": [
                    {
                        "type": "check_scale_consistency",
                        "reason": "Ensure consistent scale relationships between objects",
                        "tolerance": 0.3
                    },
                    {
                        "type": "custom_operation",
                        "reason": "Prioritize objects with known real-world dimensions",
                        "operation": {
                            "type": "filter_by_known_dimensions"
                        }
                    },
                    {
                        "type": "balance_object_types",
                        "reason": "Ensure diverse object types are represented",
                        "min_per_type": 1,
                        "max_total": 5
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
            
            # 创建包含所有必要测量值的字典
            measurements = {
                'width_pixel': obj.get('width_pixel', 0),
                'height_pixel': obj.get('height_pixel', 0),
                'area_pixel': obj.get('area_pixel', 0)
            }
            
            # 确保添加所有真实世界尺寸
            measurements['width_m'] = obj.get('width_m', 1.0)  # 使用默认值如果缺失
            measurements['length_m'] = obj.get('length_m', 2.0)  # 使用默认值如果缺失
            measurements['area_m'] = obj.get('area_m', 2.0)  # 使用默认值如果缺失
            
            final_output[obj_name] = measurements

        # 验证所有对象都有必要的字段
        for obj_name, obj_data in final_output.items():
            missing_fields = []
            for field in ['width_m', 'length_m', 'area_m']:
                if field not in obj_data:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"Warning: Object {obj_name} is missing fields: {', '.join(missing_fields)}")
                # 为缺失字段添加默认值
                for field in missing_fields:
                    if field == 'width_m':
                        obj_data[field] = 1.0
                    elif field == 'length_m':
                        obj_data[field] = 2.0
                    elif field == 'area_m':
                        obj_data[field] = 2.0

        # Convert NumPy types to Python types for JSON serialization
        final_output = self._convert_numpy_types(final_output)

        try:
            # Find selected objects in the original list for visualization
            selected_obj_names = list(final_output.keys())
            selected_objects_for_vis = [obj for obj in reference_objects if obj.get('obj', '') in selected_obj_names]
            
            # Visualize selected reference objects
            if selected_objects_for_vis:
                visualization_path = os.path.join(os.path.dirname(image_path), "selected_reference_objects.jpg")
                self.sam.visualize_multiple_objects(image_path, selected_objects_for_vis, save_path=visualization_path)
                print(f"Selected reference objects visualization saved to {visualization_path}")
        except Exception as e:
            print(f"Warning: Could not visualize selected reference objects: {e}")
        
        return final_output


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = ReferenceDetectionAgent()
    
    # Run the complete workflow
    image_path = "/Users/wangyinghao/Desktop/spatial_query_reasoning_agent/demo_P0038.png"
    query = "What is scale of image, i.e. how many meters per pixel"
    
    result = agent.run(image_path, query)
    print(f"Final result: {json.dumps(result, indent=2)}")
import os
import json
from openai import OpenAI

# Import API functions - these are defined in a API_reference_obj.py
# Agent can dynamically call these functions
from API_reference_obj import (
    call_geochat_for_image_caption,
    generate_reference_list
    select_optimal_references,
    call_geochat_for_segmentation,
    calculate_pixel_metrics,
    call_unidepth,
    infer_scale_factor,
    calculate_real_world_metrics,
)


class SpatialMetricQueryAgent:
    """
    Dynamic agent for spatial metric query answering in remote sensing images.
    Dynamically calls external API functions to perform analysis.
    """
    
    def __init__(self, openai_api_key=None):
        """Initialize Spatial Metric Query Agent"""
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def transform_query(self, query: str, image_path: str) -> str:
        """
        Transform the original query into a form that's suitable for spatial metric analysis
        """
        
        system_prompt = """You are a query transformation assistant for remote sensing image analysis.

Your task is to transform general questions about spatial metrics into specific queries about measuring objects in remote sensing images.

For example:
- "How big is that building?" → "What is the building's real-world footprint area in square meters?"
- "What's the length of the road?" → "What is the real-world length of the visible road segment in meters?"
- "How far apart are these houses?" → "What is the real-world distance between the two houses in meters?"

Important rules:
1. Always transform queries to focus on specific spatial measurements
2. Include the appropriate unit of measurement (meters, square meters, etc.)
3. Clarify whether the measurement is for length, area, perimeter, or distance
4. Make it clear the analysis is based on the remote sensing image
5. Keep the transformed query concise and clear

Return only the transformed query without any explanation or additional text."""

        user_prompt = f"""Original query: {query}

Transform this query to focus specifically on spatial metric measurements in a remote sensing image."""

        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            transformed_query = completion.choices[0].message.content.strip()
            return transformed_query
        except Exception as e:
            print(f"Error transforming query: {e}")
            # Return original query if transformation fails
            return query
    
    def design_analysis_plan(self, query: str, image_path: str) -> Dict:
        """Design spatial analysis plan based on query"""
        system_prompt = """You are a spatial analysis assistant that helps design plans for measuring objects in remote sensing images.
You can use the following functions:

1. "call_geochat_for_image_caption": Generate a comprehensive caption/description of the remote sensing image
2. "extract_reference_objects_from_caption": Extract potential reference objects from the image caption
3. "select_optimal_references": Select the most reliable reference objects for scale estimation
4. "segment_objects_with_sam2": Segment objects in the image using SAM2
5. "calculate_pixel_metrics": Calculate pixel-level metrics (area, length, etc.)
6. "infer_scale_factor": Infer the scale factor based on reference objects
7. "calculate_real_world_metrics": Calculate real-world measurements

Please return an execution plan in JSON format:
{
  "plan": [
    {"function": "function_name", "description": "purpose of this step"},
    {"function": "another_function", "description": "purpose of this step"}
  ],
  "explanation": "explanation of the execution plan",
  "target_metric": "area/length/distance/perimeter"
}

Note:
- Reference objects are essential for calculating the scale factor
- The plan should include the identification of both the reference objects and the object of interest
- Different metrics (area, length, etc.) may require different reference objects and calculation methods
- Only select necessary functions required to complete the task"""

        user_prompt = f"Query: {query}\nImage path: {image_path}\nPlease design a spatial analysis plan."

        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            plan = json.loads(completion.choices[0].message.content)
            print(f"Analysis plan: {json.dumps(plan, indent=2, ensure_ascii=False)}")
            return plan
        except Exception as e:
            print(f"Error designing analysis plan: {e}")
            return {"plan": [], "explanation": f"Error designing analysis plan: {e}"}
    
    def execute_plan(self, plan: Dict, image_path: str) -> Dict:
        """Execute spatial analysis plan by dynamically calling the appropriate API functions"""
        
        # Dictionary to store intermediate results
        data = {
            "image_path": image_path,
            "target_metric": plan.get("target_metric", "unknown")
        }
        
        # Track completed steps
        for step_idx, step in enumerate(plan.get("plan", [])):
            function_name = step.get("function")
            description = step.get("description", "No description")
            
            try:
                # Call the appropriate function based on the plan
                # This demonstrates the dynamic nature of the agent
                
                if function_name == "call_geochat_for_image_caption":
                    caption = call_geochat_for_image_caption(
                        self.openai_api_key, 
                        image_path
                    )
                    data["image_caption"] = caption
                
                elif function_name == "extract_reference_objects_from_caption":
                    if "image_caption" not in data:
                        print("No image caption available, skipping reference extraction")
                        continue
                        
                    reference_objects = extract_reference_objects_from_caption(
                        self.openai_api_key,
                        data["image_caption"],
                        image_path
                    )
                    data["reference_objects"] = reference_objects
                
                elif function_name == "select_optimal_references":
                    if "reference_objects" not in data:
                        print("No reference objects detected, skipping selection")
                        continue
                        
                    data["selected_references"] = select_optimal_references(
                        self.openai_api_key,
                        data["reference_objects"], 
                        data["target_metric"]
                    )
                
                
                elif function_name == "calculate_pixel_metrics":
                    if "segmentation_results" not in data:
                        print("No segmentation results, skipping pixel metrics")
                        continue
                        
                    data["pixel_metrics"] = calculate_pixel_metrics(
                        data["segmentation_results"],
                        data["target_metric"]
                    )
                
                elif function_name == "infer_scale_factor":
                    if "pixel_metrics" not in data or "selected_references" not in data:
                        print("Missing required data, skipping scale factor inference")
                        continue
                        
                    data["scale_factor"] = infer_scale_factor(
                        data["selected_references"],
                        data["pixel_metrics"]
                    )
                
                elif function_name == "calculate_real_world_metrics":
                    if "scale_factor" not in data or "pixel_metrics" not in data:
                        print("Missing required data, skipping real-world metrics")
                        continue
                        
                    data["real_world_metrics"] = calculate_real_world_metrics(
                        data["pixel_metrics"],
                        data["scale_factor"],
                        data["target_metric"]
                    )
                
                else:
                    print(f"Unknown function: {function_name}")
            
            except Exception as e:
                print(f"Error executing {function_name}: {e}")
                data[f"{function_name}_error"] = str(e)
        
        return data
    
    def process_query(self, query: str, image_path: str) -> str:
        """Main function to process spatial metric query"""
        
        if not os.path.exists(image_path):
            return f"Error: Image file does not exist: {image_path}"
        
        try:
            # Store original query
            original_query = query
            
            # 1. Transform the query
            transformed_query = self.transform_query(query, image_path)
            
            # 2. Design analysis plan based on transformed query
            plan = self.design_analysis_plan(transformed_query, image_path)
            
            # 3. Execute plan
            results = self.execute_plan(plan, image_path)
            
            # 4. Generate response using the API function
            response = generate_spatial_response(
                self.openai_api_key,
                transformed_query, 
                results, 
                original_query
            )
            
            return response
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(f"\n {error_message}")
            import traceback
            traceback.print_exc()
            return error_message


if __name__ == "__main__":
    # Initialize Agent
    agent = SpatialMetricQueryAgent()
    
    # Example usage
    image_path = "..."  # Path to a remote sensing image
    
    # Get user query
    query = input("\nEnter your spatial query: ")
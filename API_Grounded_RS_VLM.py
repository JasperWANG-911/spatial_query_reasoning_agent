import os
import json
import base64
from typing import List, Dict, Any
from openai import OpenAI
import requests

# Predefined reference object list with known dimensions
reference_object_list = [
    {
        "type": "car",
        "dimensions": {"length": 4.5, "width": 1.8},
        "area": 8.1,
        "reliability": 0.95
    },
    {
        "type": "bus",
        "dimensions": {"length": 12.0, "width": 2.5},
        "area": 30.0,
        "reliability": 0.9
    },
    {
        "type": "truck",
        "dimensions": {"length": 16.0, "width": 2.5},
        "area": 40.0,
        "reliability": 0.9
    },
    {
        "type": "tennis_court",
        "dimensions": {"length": 23.77, "width": 10.97},
        "area": 260.76,
        "reliability": 0.95
    },
    {
        "type": "basketball_court",
        "dimensions": {"length": 28.0, "width": 15.0},
        "area": 420.0,
        "reliability": 0.9
    },
    {
        "type": "soccer_field",
        "dimensions": {"length": 105.0, "width": 68.0},
        "area": 7140.0,
        "reliability": 0.9
    },
    {
        "type": "olympic_pool",
        "dimensions": {"length": 50.0, "width": 25.0},
        "area": 1250.0,
        "reliability": 0.7
    },
    {
        "type": "boeing_737",
        "dimensions": {"length": 39.5, "width": 35.8},
        "area": 285.46,
        "reliability": 0.9
    },
    {
        "type": "boeing_787",
        "dimensions": {"length": 56.7, "width": 60.1},
        "area": 690.88,
        "reliability": 0.9
    },
    {
        "type": "airbus_320",
        "dimensions": {"length": 37.57, "width": 35.8},
        "area": 282.36,
        "reliability": 0.9
    },
    {
        "type": "airbus_330",
        "dimensions": {"length": 63.69, "width": 60.3},
        "area": 764.51,
        "reliability": 0.9
    },
    {
    "type": "airbus_321",
    "dimensions": {"length": 44.51, "width": 35.8},
    "area": 334.3,
    "reliability": 0.9
}
]

def analyse_caption_for_references(caption: str) -> List[str]:
    """Analyze image caption to identify reference objects that match our predefined list."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    reference_types = [obj["type"] for obj in reference_object_list]

    user_prompt = f"""
    TASK: Carefully analyze the following caption and identify ANY objects that semantically match items in the reference list.

    CAPTION:
    "{caption}"

    REFERENCE OBJECTS:
    {', '.join(reference_types)}

    INSTRUCTIONS:
    1. Find ALL objects in the caption that match OR are synonymous with items in the reference list
    2. Consider synonyms and related terms (e.g., "automobile" → "car", "boeing 737" → "boeing_737")
    3. Be particularly careful with aircraft identification:
       - "boeing 737", "b737", "boeing737" should match "boeing_737"
       - "boeing 787", "b787", "boeing787" should match "boeing_787"
       - "airbus 320", "a320", "airbus320" should match "airbus_320" (if in the reference list)
    4. Look for contextual clues (e.g., "vehicles parked" likely refers to "car")
    5. Be thorough - missing matches will affect downstream analysis
    6. Use exact names from the reference list in your response
    7. If the caption mentions planes or aircraft without specific models, do your best to infer the
       most likely aircraft type from the reference list based on the context
    """

    SYSTEM_PROMPT = """
    You are an expert object matching system specialized in identifying references to objects in text descriptions.

    CRITICAL GUIDELINES:
    - Be COMPREHENSIVE - identify ALL potential matches between the caption and reference list
    - Use SEMANTIC matching (e.g., "automobile", "vehicle", "sedan" should match "car")
    - Match PLURAL forms to singular (e.g., "cars" → "car") 
    - Match SPECIFIC types to general categories when appropriate
    - Return matches using EXACTLY the names from the reference list
    - Even VAGUE references should be matched if reasonably confident
    - Pay SPECIAL ATTENTION to aircraft models - match any mention of specific aircraft to the correct reference model
    - Return your results EXCLUSIVELY as a JSON object: {"objects": ["type1", "type2"]}

    BE THOROUGH - it's better to include more potential matches than to miss important ones.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3  # Lower temperature for more consistent matching
    )

    try:
        result = json.loads(response.choices[0].message.content)
        matched_objects = result.get("objects", [])
        print(f"Raw GPT-4o response for reference matching: {response.choices[0].message.content}")
        return matched_objects
    except Exception as e:
        print(f"Error parsing reference matching response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return []

class GeoChat:
    """GeoChat class reimplemented using GPT-4o with functionality including image captioning and returning bbox."""
    
    def __init__(self, api_key=None):
        """Initialize GeoChat with OpenAI API key"""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an environment variable or pass it as an argument.")
        self.client = OpenAI(api_key=self.api_key)
    
    def _encode_image(self, image_path):
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_captionning(self, image_path: str) -> str:
        """Generate image caption using GPT-4o vision capabilities with enhanced aircraft identification"""
        # Encode image for API
        base64_image = self._encode_image(image_path)
        
        # Enhanced system prompt for detailed captioning with specific aircraft identification
        system_prompt = """
        You are a geospatial image analysis expert specialized in remote sensing imagery. 
        Provide detailed descriptions of the image, focusing on objects, infrastructure, and landscape features.
        
        CRITICAL INSTRUCTIONS FOR AIRCRAFT IDENTIFICATION:
        - If you see ANY aircraft in the image, you MUST identify its specific model.
        - For Boeing aircraft, specifically identify as "boeing_737" or "boeing_787" rather than generic terms.
        - For Airbus aircraft, specifically identify as "airbus_320", "airbus_321", or "airbus_330" rather than generic terms.
        - Do NOT use generic terms like "airplane", "aircraft", or "plane" without specifying the model.
        - If you cannot determine the exact model, make your best estimation based on visible features like size, shape, wingspan, etc.
        - When mentioning aircraft, always include the manufacturer and model in the format: "manufacturer_model" (e.g., "boeing_737", "airbus_320").
        
        Include in your description:
        1. Main objects and structures visible in the image
        2. Spatial relationships between objects
        3. Context and setting (urban, rural, industrial, etc.)
        
        Be thorough but concise. Focus on factual observations over interpretations.
        """
        
        # User prompt that specifically asks for aircraft identification if present
        user_prompt = """
        Provide a detailed description of this image. If you see any aircraft, be sure to identify the specific model (like boeing_737, boeing_787, airbus_320, etc.) rather than using generic terms like "plane" or "aircraft".
        """
        
        # Call GPT-4o Vision API
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        
        # Double-check for generic aircraft terms and follow up if needed
        generic_aircraft_terms = ["aircraft", "airplane", "plane", "jet"]
        specific_aircraft_models = ["boeing_737", "boeing_787", "airbus_320", "airbus_321", "airbus_330"]
        
        has_generic_terms = any(term in description.lower() for term in generic_aircraft_terms)
        has_specific_models = any(model in description.lower() for model in specific_aircraft_models)
        
        # If it mentions aircraft generically but doesn't specify models, follow up with a specific question
        if has_generic_terms and not has_specific_models:
            follow_up_prompt = """
            I noticed you mentioned aircraft in your description, but did not specify the exact model. 
            Please analyze the image again and identify if the aircraft is boeing_737, boeing_787, airbus_320, airbus_321, 
            airbus_330, or another specific model. This is critical for scale calculation purposes.
            """
            
            follow_up_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": description
                    },
                    {
                        "role": "user",
                        "content": follow_up_prompt
                    }
                ],
                max_tokens=200
            )
            
            # Append the specific aircraft identification to the original description
            aircraft_identification = follow_up_response.choices[0].message.content
            description = f"{description}\n\nAIRCRAFT IDENTIFICATION: {aircraft_identification}"
        
        return description
    
    def generate_bbox(self, image_path: str, object_list: List[str]) -> List[Dict[str, Any]]:
        """Generate bounding boxes for objects in the image using LandingAI's API"""
        if not object_list:
            return []
        
        print(f"Attempting to generate bounding boxes for: {object_list}")
        
        # Initialize result list
        result_boxes = []
        
        # Process each object type
        for obj_type in object_list:
            # Call LandingAI API
            AOD_url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
            headers = {"Authorization": "Basic ejRkbG43a2RsaGxndnF5ZWdpbGp1Om55S1ZoWVMydlJ6QkJxSGp5Z2plQ3ZkeG42a1RmNVhi"}
            
            try:
                with open(image_path, "rb") as image_file:
                    files = {"image": image_file}
                    data = {"prompts": obj_type, "model": "agentic"}
                    response = requests.post(AOD_url, files=files, data=data, headers=headers)
                    
                    aod_result = response.json()
                    print(f"API response for {obj_type}: {aod_result}")
                    
                    # Extract detections - handle the specific structure from LandingAI
                    if "data" in aod_result and aod_result["data"] and len(aod_result["data"]) > 0:
                        # The first element of the "data" list contains the detections
                        detections = aod_result["data"][0]
                        
                        for i, detection in enumerate(detections):
                            if "bounding_box" in detection and detection["label"].lower() == obj_type.lower():
                                # Extract coordinates
                                bbox = detection["bounding_box"]
                                
                                # Add to results with proper naming for multiple instances
                                result_boxes.append({
                                    "obj_name": f"{obj_type}_{i+1}" if len(detections) > 1 else obj_type,
                                    "bbox": [int(coord) for coord in bbox]  # Ensure all values are integers
                                })
            
            except Exception as e:
                print(f"Error detecting {obj_type}: {e}")
        
        print(f"Generated {len(result_boxes)} bounding boxes")
        return result_boxes
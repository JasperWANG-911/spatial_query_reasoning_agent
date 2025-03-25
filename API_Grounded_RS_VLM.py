"""
API for Grounded RS VLM
This module provides a fixed GeoChat class for generating image descriptions and detecting objects.
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI

# Predefined reference object list with known dimensions(to be modified)
# Also, my idea is pre-defining this list is better than generating 
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
        "dimensions": {"length": 16.0, "width": 2.5, "height": 4.0},
        "area": 40.0,
        "reliability": 0.85
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
        "reliability": 0.95
    },
    {
        "type": "soccer_field",
        "dimensions": {"length": 105.0, "width": 68.0},
        "area": 7140.0,
        "reliability": 0.95
    },
    {
        "type": "olympic_pool",
        "dimensions": {"length": 50.0, "width": 25.0},
        "area": 1250.0,
        "reliability": 0.95
    },
    {
        "type": "road_lane",
        "dimensions": {"width": 3.5},
        "reliability": 0.8
    },
    {
        "type": "parking_space",
        "dimensions": {"length": 5.5, "width": 2.5},
        "area": 13.75,
        "reliability": 0.85
    },
    {
        "type": "airplane_737",
        "dimensions": {"length": 33.6, "width": 35.8},  # width includes wingspan
        "area": 1202.88,
        "reliability": 0.98
    },
    {
        "type": "airplane_747",
        "dimensions": {"length": 70.6, "width": 59.6},  # width includes wingspan
        "area": 4207.76,
        "reliability": 0.98
    },
]

def analyse_caption_for_references(caption: str) -> List[str]:
    """
    Analyze image caption to identify reference objects that match our predefined list.
    Return a jason list in forat {"objects":["A", "B"]}
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    reference_types = [obj["type"] for obj in reference_object_list]

    user_prompt = f"""
    Analyze the following image caption and identify any objects that match the reference list below.
    
    Caption: "{caption}"

    Reference objects to look for: {', '.join(reference_types)}
    """

    SYSTEM_PROMPT = """
    You are a specialized text-based object matching assistant. Your ONLY task is to identify references
    to objects from a given caption that EXACTLY match the provided reference list. IMPORTANT RULES:
    (1) Always perform semantic matching. (2) Always return matched objects using the EXACT names from the 
    provided reference list, NEVER variations from the caption. (3) Include ALL reasonably confident matches;
    exclude uncertain or unclear matches. (4) Return your results exclusively as a JSON object in the following 
    format: {"objects": ["type1", "type2"]}. DO NOT add explanations or extra text.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("objects", [])

# For Cheng: we need to complete this class by integrating geochat into this repo
class GeoChat:
    """
    GeoChat class with functionality including image captioning and returning bbox.
    """
    
    def __init__(self):
    
    def generate_captionning(self, image_path: str) -> str: # generate image caption
    
    def generate_bbox(self, image_path: str) -> List[Dict[str, Any]]: # use list generated from analyse_caption_for_references to generate a dictionary {'obj_name': 'bbox'}

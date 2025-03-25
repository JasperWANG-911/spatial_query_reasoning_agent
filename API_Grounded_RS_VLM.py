"""
API for Grounded RS VLM
This module provides a fixed GeoChat class for generating image descriptions and detecting objects.
"""

import os
import base64
import json
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI
from PIL import Image

# For Cheng: we need to complete this class by integrating geochat into this repo
class GeoChat:
    """
    GeoChat class with unctionality including image captioning and returning bbox.
    """
    
    def __init__(self):
    
    def call_geochat_for_captionning(self, image_path: str) -> str:
    
    def call_geochat_for_bbox(self, image_path: str) -> List[Dict[str, Any]]:

from PIL import Image
from io import BytesIO
import base64
import asyncio
import numpy as np
import re
from ve import VE


class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.extract_image = operators['ExtractImage']
        self.image_to_url = operators['ImageToUrl']
        self.sam2 = operators['SAM2']
        self.depth = operators['Depth-Anything-V2']
        self.prompt_custom = prompt_custom
    
    async def run(self, question: list[VE | str]) -> str:
        image = question[0]
        image = self.crop(image)
        response = self.aggregate(map(self.subproblem_generation(question[1]), self.sam(image)))
        return response
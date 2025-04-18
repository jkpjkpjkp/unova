from PIL import Image
from io import BytesIO
import base64
import asyncio
import numpy as np
import re
from ve import VE
from pydantic import BaseModel

class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.extract_image = operators['ExtractImage']
        self.image_to_url = operators['ImageToUrl']
        self.sam = operators['SAM2']
        self.depth = operators['Depth-Anything-V2']
        self.prompt_custom = prompt_custom
    
    def subproblem_generation(self, question) -> str:
        class splitaggregate(BaseModel):
            subquestion: str
            aggregation: str
        return self.custom(f"""we plan to split the image for this question into parts, and we now need you to write subquestion for every part of the image.

1. please write the subquestion asked for every part of the image
2. please write an aggregation statement that combines sub parts' answers into an answer of the original question. 
""", dna = splitaggregate)
    
    async def run(self, question: tuple[VE, str]) -> str:
        image = question[0]
        image = self.crop(image)
        subquestion, aggregation = self.subproblem_generation(question[1])
        response = self.aggregate(aggregation, map(self.custom(subquestion), self.sam(image)))
        return response
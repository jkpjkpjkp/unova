from ve import VE
from pydantic import BaseModel, Field

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.crop = operators['Crop']
        self.prompts = prompts
    
    async def run(self, question: tuple[VE, str]) -> str:
        image = await self.crop(*question)
        response = await self.custom(input=(image, question[1], self.prompts['COT']))
        return response
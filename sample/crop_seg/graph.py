from ve import VE
from pydantic import BaseModel, Field

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.crop = operators['Crop']
        self.prompts = prompts
    
    def info_spread(self, mask: Image.Image, question: str):
        self.custom(self.prompts['INFO_SPREAD'].format(image=mask, question=question))
    
    async def run(self, question: tuple[VE, str]) -> str:
        image = await self.crop(*question)
        sam = image.sam()
        info = map(self.info_spread, sam, question[1])
        response = await self.info_gather(info)
        return response
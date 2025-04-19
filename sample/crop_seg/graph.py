from ve import VE
from pydantic import BaseModel, Field

class Graph:
    def __init__(self, operators: dict):
        self.custom = operators['Custom']
        self.crop = operators['Crop']
    
    def subproblem_generation(self, question) -> str:
        class splitaggregate(BaseModel):
            subquestion: str = Field(default="", description="this subquestion will be asked to every subpart we cut out of the image.")
            aggregation: str
        return self.custom(f"""we plan to split the image for this question into parts, and we now need you to write subquestion for every part of the image.

1. please write the subquestion asked for every part of the image
2. please write an aggregation statement that combines sub parts' answers into an answer of the original question. 

original question: {question}
splitting method: we are using sam2 auto mask gen, and use each mask as a split. 

""", pydantic_model = splitaggregate, mode='xml_fill')
    
    def aggregate(self, aggregation, subanswers):
        return self.custom(f"""""")
        
    
    async def run(self, question: tuple[VE, str]) -> str:
        image = question[0]
        image = await self.crop(*question)
        sub = await self.subproblem_generation(question[1])

        response = self.aggregate(sub['aggregation'], map(self.custom(sub['subquestion']), self.image.sam()))
        return response
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
    
    async def aggregate(self, aggregation, subanswers):
        subanswers_str = ', '.join(subanswers) 
        prompt = f"Given the following answers from different parts of the image: [{subanswers_str}], please {aggregation} to get the final answer."
        return await self.custom(prompt)
        
    
    async def run(self, question: tuple[VE, str]) -> str:
        image = question[0]
        image = await self.crop(*question)
        sub = await self.subproblem_generation(question[1])

        response = await self.aggregate(sub['aggregation'], list(map(self.custom(sub['subquestion']), image.sam())))
        return response
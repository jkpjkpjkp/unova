from PIL import Image
from db import MyStr

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.som = operators['SoM']
        self.prompts = prompts
    
    async def run(self, question: tuple[Image.Image, str]) -> str:
        print("Debug: Starting run method")
        print(f"Debug: Question text: {question[1]}")
        print(f"Debug: Input image type: {type(question[0])}")
        
        image = question[0]
        print("Debug: Calling SoM operator")
        set_of_mask = await self.som(image)
        print(f"Debug: SoM output type: {type(set_of_mask)}")
        
        formatted_prompt = self.prompts['COT'].format(
            original_image=image,
            set_of_mask_image=set_of_mask,
            question=question[1]
        )
        print(f"Debug: Formatted prompt: {formatted_prompt[:200]}...")  # First 200 chars
        assert isinstance(self.prompts['COT'], MyStr)
        assert isinstance(formatted_prompt, tuple)
        print("Debug: Calling custom operator")
        response = await self.custom(input=formatted_prompt)
        print(f"Debug: Custom operator response: {response}")
        
        return response['response']
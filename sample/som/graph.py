from PIL import Image
class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.som = operators['SoM']
        self.prompts = prompts
    
    async def run(self, question: tuple[Image.Image, str]) -> str:
        image = question[0]
        set_of_mask = await self.som(image)
        response = await self.custom(input=self.prompts['COT'].format(original_image=image, set_of_mask_image=set_of_mask, question=question[1]))
        return response['response']
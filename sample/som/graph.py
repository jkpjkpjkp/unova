from PIL import Image
class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.som = operators['SoM']
        self.prompts = prompts
    
    async def run(self, question: tuple[Image.Image, str]) -> str:
        image = await self.som(question[0])
        response = await self.custom(input=self.prompts['COT'].format(image=image, question=question[1]))
        return response['response']
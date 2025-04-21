from PIL import Image

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.prompts = prompts

    async def run(self, problem: tuple[Image.Image, str]) -> str:
        return (await self.custom(input=self.prompts['COT'].format(image=problem[0], question=problem[1])))['response']

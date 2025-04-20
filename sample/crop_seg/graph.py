from PIL import Image

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.crop = operators['Crop']
        self.sam = operators['SAM']
        self.som = operators['SoM']
        self.prompts = prompts
    
    def info_spread(self, mask: Image.Image, question: str):
        self.custom(self.prompts['INFO_SPREAD'].format(image=mask, question=question))
    
    def info_gather(self, info: list[str], som_image):
        self.custom(self.prompts['INFO_GATHER'].format(parts='\n'.join(info), SoM_image=som_image))

    async def run(self, question: tuple[Image.Image, str]) -> str:
        # image = await self.crop(*question)
        image = question[0]
        sam = self.sam(image)
        som = self.som(sam)
        info = map(self.info_spread, sam, question[1])
        response = await self.info_gather(info, som)
        return response
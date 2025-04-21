from PIL import Image

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.crop = operators['Crop']
        self.sam = operators['SAM']
        self.som = operators['SoM']
        self.prompts = prompts
    
    async def info_spread(self, mask: Image.Image, question: str):
        return (await self.custom(self.prompts['INFO_SPREAD'].format(image=mask, question=question)))['response']
    
    async def info_gather(self, question: str, info: list[str], som_image):
        info = [x for x in info if x]
        assert info
        return (await self.custom(self.prompts['INFO_GATHER'].format(question=question, parts='\n'.join(info), SoM_image=som_image)))['response']

    async def run(self, question: tuple[Image.Image, str]) -> str:
        # image = await self.crop(*question)
        image = question[0]
        sam = self.sam(image)
        assert len(sam)
        som = self.som(sam)
        assert isinstance(som, Image.Image), som
        assert som.getbbox()
        info = [await self.info_spread(x, question[1]) for x in sam]
        response = await self.info_gather(question[1], info, som)
        return response
from PIL import Image
from io import BytesIO
import base64
import asyncio
import numpy as np
import re



class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.extract_image = operators['ExtractImage']
        self.image_to_url = operators['ImageToUrl']
        self.sam2 = operators['SAM2']
        self.depth = operators['Depth-Anything-V2']
        self.prompt_custom = prompt_custom
    
    def extract_brace(self, x: str):
        import re
        match = re.search(r"{(.*?)}", x)
        if not match:
            return None
        return match.group(1)

    async def bbox_mask_infos(self, image, depth, bbox):
        bbox = (bbox[0] / 1000 * image.height, bbox[1] / 1000 * image.width, bbox[2] / 1000 * image.height, bbox[3] / 1000 * image.width)
        image = image.crop(bbox)
        masks = await self.sam2(image)
        captions = await asyncio.gather(*[self.mask_caption(image, mask) for mask in masks])
        bboxes = [mask['bbox'] for mask in masks]

        centers = [np.average((x, y, depth[x, y]) for x, y in np.argwhere(mask['mask'])) for mask in masks]
        return [(caption, center) for caption, center in zip(captions, centers)]

    async def run(self, question: str) -> str:
        bbox = await self.custom(input=self.prompt_custom['BBOX'].format(question=question), model='gemini-2.5-pro-exp-03-25')
        image = await self.extract_image(question)[0]
        image = image.crop(bbox)
        try:
            bboxes = [tuple(int(x.strip()) for x in part.split(',')) for part in re.findall(r"{(.*?)}", bbox)]
        except:
            bboxes = [tuple(int(x.strip()) for x in part.split(',')) for part in re.findall(r"\[(.*?)\]", bbox)]
        
        depth = await self.depth(image)
        mask_info = []
        for bbox in bboxes:
            mask_info.extend(await self.bbox_mask_infos(image, depth, bbox))

        

        final_answer = await self.custom(input=self.prompt_custom['AUGMENTATION_EXPLENATION'].format(question=question, masks=mask_info), model='gemini-2.5-pro-exp-03-25')

        return final_answer
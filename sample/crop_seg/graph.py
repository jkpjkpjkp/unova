from PIL import Image
from io import BytesIO
import base64
import asyncio
import numpy as np
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
    
    async def mask_caption(self, image, mask):
        mask, bbox = mask['mask'], mask['bbox']
        masked = image.copy()

        alpha_mask = Image.fromarray(mask * 255, mode='L')
        masked.putalpha(alpha_mask)

        cropping_cascade = [masked]

        center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        height = (bbox[2] - bbox[0]) // 2
        width = (bbox[3] - bbox[1]) // 2
        for _ in range(3):
            bbox = [max(0, center[0] - height), max(0, center[1] - width), min(image.height, center[0] + height), min(image.width, center[1] + width)]
            cropped_img = image.crop(bbox)
            cropped_img.thumbnail((512, 512))
            cropping_cascade.append(cropped_img)
            if bbox == [0, 0, image.height, image.width]:
                break
            width *= 2
            height *= 2

        response = await self.custom(
            input=self.prompt_custom['MASK_CAPTION'] + ' '.join(self.image_to_url(cropped) for cropped in cropping_cascade),
            model='gemini-2.0-flash',
        )

        return self.extract_brace(response) or response



    async def run(self, question: str) -> str:
        bbox = await self.custom(input=self.prompt_custom['BBOX'].format(question=question), model='gemini-2.5-pro-exp-03-25')
        image = await self.extract_image(question)
        image = image.crop(bbox)
        masks = await self.sam2(image)
        bboxes = [mask['bbox'] for mask in masks]

        centers = [np.average((x, y, depth[x, y]) for x, y in np.argwhere(mask['mask'])) for mask in masks]

        captions = await asyncio.gather(*[self.mask_caption(image, mask) for mask in masks])
        depth = await self.depth(image)
        mask_info = [(caption, center) for caption, center in zip(captions, centers)]

        final_answer = await self.custom(input=self.prompt_custom['AUGMENTATION_EXPLENATION'].format(question=question, masks=mask_info), model='gemini-2.5-pro-exp-03-25')

        return final_answer
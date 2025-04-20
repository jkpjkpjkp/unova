from typing import Iterator
from PIL import Image
import functools
import numpy as np

def center(img: Image.Image):
    return np.average(np.where(img.to_numpy()))

class VisualEntity:
    _img: Image.Image | list['VisualEntity']

    def __init__(self, img: Image.Image | list['VisualEntity']):
        if isinstance(img, VE):
            self = img
        else:
            self._img = img
    def __iter__(self) -> Iterator['VisualEntity']:
        if isinstance(self._img, Image.Image):
            yield self
        else:
            return iter(self._img)
    def __len__(self):
        return len(self._img) if isinstance(self._img, list) else 1
    def __getitem__(self, item) -> 'VisualEntity':
        if isinstance(self._img, Image.Image):
            assert item == 0
            return self
        else:
            return self._img[item]
    def __add__(self, other: 'VisualEntity') -> 'VisualEntity':
        l = self._img if isinstance(self._img, list) else [self]
        r = other._img if isinstance(other._img, list) else [other]
        return VisualEntity(l + r)
    
    @property
    def center(self):
        return np.average(np.where(self._img.to_numpy()))
    @property
    def area(self) -> int:
        return sum(self.image.getdata(band=3) / 255)
    @property
    def shape(self):
        img = self._img if isinstance(self._img, Image.Image) else self._img[0]
        return img.width, img.height
    
    def crop(self, xyxy: tuple[int, int, int, int] | None = None):
        assert isinstance(self._img, Image.Image)
        return self._img.crop(xyxy or self.bbox)
    def crop1000(self, box: tuple):
        x, y = self.shape
        return VE(self.crop((box[0] / 1000 * x, box[1] / 1000 * y, box[2] / 1000 * x, box[3] / 1000 * y)))
    
    @property
    def image(self):
        return self._img if isinstance(self._img, Image.Image) else functools.reduce(lambda a, b: a.alpha_composite(b), self._img, initial=Image.new('RGBA', self._img[0].size, (0, 0, 0, 0)))
    @property
    def bbox(self):
        return self.image.getbbox()
    def present(self, mode='raw') -> list[Image.Image]:
        if mode == 'raw':
            return [self.crop()]
        elif mode == 'box':
            return [self.crop().to('RGB')]
        elif mode == 'cascade':
            center = tuple(int, int)(self.center())
            box = self.bbox
            x, y = box // 2
            return VE([self._img.crop(xyxy=(center[0]-x*2**i, center[1]-y*2**i, center[0]+x*2**i, center[1]+y*2**i)).thumbnail(self.bbox) for i in range(3)])
        elif mode == 'number':
            from PIL import ImageDraw, ImageFont
            def draw_number(number: int, image: Image.Image) -> Image.Image:
                img = image.copy()
                draw = ImageDraw.Draw(img)
                bbox = img.getbbox()
                if bbox:  # Only draw if there’s a non-transparent region
                    x = (bbox[0] + bbox[2]) / 2
                    y = (bbox[1] + bbox[3]) / 2
                    font = ImageFont.load_default()
                    text = str(number)
                    draw.text((x, y), text, fill='red', font=font, anchor='mm')  # Center the text
                return img
            return [draw_number(i, ve.image) for i, ve in enumerate(self)]


    def sam(self):
        return VE(sam2_imagecrop(self._img))

VE = VisualEntity
if __name__ == '__main__':

    from zero import get_task_data
    def test_number_render():
        image, _ = get_task_data('42_2')
        ve = VE(image)
        assert isinstance(ve.sam().present('number')[0], Image.Image)
    
    test_number_render()
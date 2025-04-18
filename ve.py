import re
from PIL import Image
from typing import Literal, Self, Iterator
import functools
import numpy as np

def area(img: Image.Image) -> int:
    R,G,B,A = tuple(range(4))
    return sum(img.getdata(band=A) / 255)

class VisualEntity:
    _img: Image.Image | list['VisualEntity']
    

    def __init__(self, img: Image.Image | list[Self]):
        self._img = img
    def __iter__(self) -> Iterator[Self]:
        if isinstance(self._img, Image.Image):
            yield self
        else:
            return iter(self._img)
    def __len__(self):
        return len(self._img) if isinstance(self._img, list) else 1
    def __getitem__(self, item) -> Self:
        if isinstance(self._img, Image.Image):
            assert item == 0
            return self
        else:
            return self._img[item]
    def __add__(self, other: Self) -> Self:
        l = self._img if isinstance(self._img, list) else [self]
        r = other._img if isinstance(other._img, list) else [other]
        return VisualEntity(l + r)
    
    def center(self):
        return np.average(np.where(self._img.to_numpy()))
    def crop(self, xyxy: tuple[int, int, int, int] | None = None):
        assert isinstance(self._img, Image.Image)
        return Self(self._img.crop(xyxy or self.bbox))
    
    @property
    def image(self):
        return self._img if isinstance(self._img, Image.Image) else functools.reduce(lambda a, b: a.alpha_composite(b), self._img, initial=Image.new('RGBA', self._img[0].size, (0, 0, 0, 0)))
    @property
    def bbox(self):
        return self.image.getbbox()
    def present(self, mode='raw'):
        if mode == 'raw':
            return self.crop(self.image)
        elif mode == 'box':
            return self.crop(self.image).to('RGB')
        elif mode == 'cascade':
            center = tuple(int, int)(self.center())
            x, y = self.bbox // 2
            for _ in range(3):
                

VE = VisualEntity
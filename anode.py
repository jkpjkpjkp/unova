import re
from PIL import Image
from typing import Literal, Self, Iterator
import functools

def crop(img: Image.Image, xyxy: tuple[int, int, int, int] | None = None) -> Image.Image:
    return img.crop(xyxy or img.getbbox())
def area(img: Image.Image) -> int:
    R,G,B,A = tuple(range(4))
    return sum(img.getdata(band=A) / 255)

class VisualEntity:
    image: Image.Image | list['VisualEntity']

    def __iter__(self) -> Iterator[Self]:
        if isinstance(self.image, Image.Image):
            yield self
        else:
            return iter(self.image)
    def __len__(self):
        return len(self.image) if isinstance(self.image, list) else 1
    def __getitem__(self, item) -> Self:
        if isinstance(self.image, Image.Image):
            assert item == 0
            return self
        else:
            return self.image[item]
    def __add__(self, other: Self) -> Self:
        l = self.image if isinstance(self.image, list) else [self]
        r = other.image if isinstance(other.image, list) else [other]
        return VisualEntity(l + r)
    def aggregate(self):
        return functools.reduce(lambda a, b: a.alpha_composite(b), self.image, initial=Image.new('RGBA', self.image[0].size, (0, 0, 0, 0)))
    def render(self):
        return crop(self.aggregate())
    
    def __init__(self, img: Image.Image | list[Self]):
        self.image = img
VE = VisualEntity

def xml_hint(field_names: list[str]):
    examples = []
    for field_name in field_names:
        examples.append(f"<{field_name}>content</{field_name}>")
    example_str = "\n".join(examples)
    return f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""

def xml_extract(content: str, field_names: list[str], field_types: dict[str, type]) -> dict[str, any]:
    extracted_data: dict[str, any] = {}

    for field_name in field_names:
        pattern = rf"<{field_name}>(.*?)</{field_name}>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            raw_value = match.group(1).strip()
            field_type = field_types.get(field_name)

            if field_type == str:
                extracted_data[field_name] = raw_value
            elif field_type == int:
                try:
                    extracted_data[field_name] = int(raw_value)
                except ValueError:
                    extracted_data[field_name] = 0
            elif field_type == bool:
                extracted_data[field_name] = raw_value.lower() in ("true", "yes", "1", "on", "True")
            elif field_type == list:
                try:
                    extracted_data[field_name] = eval(raw_value)
                    if not isinstance(extracted_data[field_name], list):
                        raise ValueError
                except:
                    extracted_data[field_name] = []
            elif field_type == dict:
                try:
                    extracted_data[field_name] = eval(raw_value)
                    if not isinstance(extracted_data[field_name], dict):
                        raise ValueError
                except:
                    extracted_data[field_name] = {}

    return extracted_data

def test_xml():
    hint = xml_hint(['graph', 'prompt'])
    print(hint)
    data = xml_extract(hint, ['graph', 'prompt'], {'graph': str, 'prompt': str})
    assert data['graph'] == data['prompt'] == 'content'

if __name__ == "__main__":
    test_xml()
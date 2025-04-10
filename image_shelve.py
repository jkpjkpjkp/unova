import shelve
from PIL import Image
import hashlib
from multiprocessing import Lock
from typing import Any, Iterable, Optional, Union, List
import base64
import binascii
import io
import re
import openai

short_hash_to_image = shelve.open('image.shelve')
long_hash_to_short_hash = shelve.open('lts.shelve')

tot = len(long_hash_to_short_hash.keys())
tot_lock = Lock()

def get_image_by_short_hash(short_hash):
    return short_hash_to_image[short_hash]

def int_to_base62(i):
    chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if i == 0:
        return '0'
    s = ''
    while i:
        s = chars[i % 62] + s
        i //= 62
    return s

def store_image(image: Image.Image):
    global tot
    long_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if long_hash not in long_hash_to_short_hash:
        with tot_lock:
            tot += 1
            local_tot = tot
        short_hash = int_to_base62(local_tot)
        long_hash_to_short_hash[long_hash] = short_hash
        short_hash_to_image[short_hash] = image
    return long_hash_to_short_hash[long_hash]

pretty = '<image_{short_hash}>'
def pretty_encode(image: Image.Image):
    return pretty.format(short_hash=store_image(image))

def store(content):
    if isinstance(content, Image.Image):
        return pretty_encode(content)
    elif isinstance(content, dict):
        return {k: store(v) for k, v in content.items()}
    elif isinstance(content, Iterable):
        return [store(item) for item in content]
    else:
        raise ValueError(f"Cannot store {type(content)}")
    

class MultimodalMessage:
    def __init__(self, content: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None):
        self.items: List[Union[str, Image.Image]] = []

        def recursively_add(content: Union[str, Image.Image, List[Union[str, Image.Image]]]):
            if not content:
                return
            if isinstance(content, Image.Image):
                self.items.append(content)
            elif isinstance(content, str):
                self.items.append(detect_image(content) or content)
            elif isinstance(content, dict):
                map(recursively_add, content.values())
            elif isinstance(content, Iterable):
                map(recursively_add, content)
        
        recursively_add(content)
    
    def __iter__(self):
        return iter(self.items)

    def __str__(self) -> str:
        return ' '.join(pretty_encode(item) if isinstance(item, Image.Image) else item for item in self)
    def __repr__(self) -> str:
        return str(self)
MM = MultimodalMessage


def detect_image(x: str) -> Optional[Image.Image]:
    try:
        bytes_data = base64.b64decode(x)
    except binascii.Error:
        bytes_data = x.encode('latin1')
    try:
        return Image.open(io.BytesIO(bytes_data))
    except IOError:
        return None

def recursively_find_image(x: Any):
    if isinstance(x, Image.Image):
        x = pretty_encode(x)
    elif isinstance(x, str):
        image = detect_image(x)
        if image:
            x = pretty_encode(image)
    elif isinstance(x, MM):
        x = str(x)
    elif isinstance(x, dict):
        map(recursively_find_image, x.values())
    elif isinstance(x, Iterable):
        map(recursively_find_image, x)


def detach_image_from_text(x: str) -> MM:
    parts = re.split(r'<image_([a-zA-Z0-9]{1,10}?)>', x)
    return MM([
        get_image_by_short_hash(x) if i % 2 == 1 else x.strip()
        for i, x in enumerate(parts)
    ])

def recursively_restore_image(x: Any):
    if isinstance(x, dict):
        map(recursively_restore_image, x.values())
    elif isinstance(x, Iterable):
        map(recursively_restore_image, x)
    elif isinstance(x, str):
        x = detach_image_from_text(x)
    return x


###### UNDONE BELOW

def call_openai(x: str):
    parts = re.split(r'<image_([a-zA-Z0-9]{1,10}?)>', x)
    content = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "text", "text": part})
        else:
            buffered = io.BytesIO()
            get_image_by_short_hash(part).save(buffered, format="PNG")
            base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"}
            })
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=content,
        response_format={"type": "json_object"},
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    return response.choices[0].message.content

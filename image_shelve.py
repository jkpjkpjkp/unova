import shelve
from PIL import Image
import hashlib
from multiprocessing import Lock
from typing import Any, Iterable, Optional, Union, List
import base64
import binascii
import io
import re
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

def store_image(image: Image.Image):
    long_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if long_hash not in long_hash_to_short_hash:
        with tot_lock:
            tot += 1
            local_tot = tot
        short_hash = int_to_base62(local_tot)
        long_hash_to_short_hash[long_hash] = short_hash
        short_hash_to_image[short_hash] = image
    return long_hash_to_short_hash[long_hash]

def get_image_by_long_hash(long_hash):
    return short_hash_to_image[long_hash_to_short_hash[long_hash]]


pretty = '<image_{short_hash}>'
def pretty_encode(image: Image.Image):
    return pretty.format(short_hash=store_image(image))

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
    if isinstance(x, dict):
        map(recursively_find_image, x.values())
    elif isinstance(x, Iterable):
        map(recursively_find_image, x)
    elif isinstance(x, Image.Image):
        x = pretty_encode(x)
    elif isinstance(x, str):
        image = detect_image(x)
        if image:
            x = pretty_encode(image)





class MultimodalMessage:
    def __init__(self, content: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None):
        self.items: List[Union[str, Image.Image]] = []

        def recursively_add(content: Union[str, Image.Image, List[Union[str, Image.Image]]]):
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

MM = MultimodalMessage


########## UNDONE BELOW
def detech_image_in_text(x: str) -> MM:
    res = re.findall(r'\<image_([a-zA-Z0-9]{1,10}?)\>', x)

def recursively_restore_image(x: Any):
    if isinstance(x, dict):
        map(recursively_restore_image, x.values())
    elif isinstance(x, Iterable):
        map(recursively_restore_image, x)
    elif isinstance(x, str):
        match = re.findall(r'\<image_([a-zA-Z0-9]{0,10}?)\>', x)
        if not match:
            return
        if len(match) == 1:
            x = get_image_by_short_hash(match[0])
        else:
            raise ValueError("Multiple images found in string")
    return x



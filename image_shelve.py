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
import os
import argparse
model = "gemini-2.0-pro-exp-02-05"

short_hash_to_image = shelve.open('image.shelve')
long_hash_to_short_hash = shelve.open('lts.shelve')

tot = len(long_hash_to_short_hash.keys())
tot_lock = Lock()

def get_image_by_short_hash(short_hash):
    return short_hash_to_image[short_hash]

def int_to_0aA(i):
    chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if i == 0:
        return '0'
    s = ''
    while i:
        s = chars[i % 62] + s
        i //= 62
    return s

def _img_go(image: Image.Image):
    global tot
    image = image.convert('RGB')
    long_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if long_hash not in long_hash_to_short_hash or not long_hash_to_short_hash[long_hash]:
        with tot_lock:
            tot += 1
            local_tot = tot
        short_hash = int_to_0aA(local_tot)
        long_hash_to_short_hash[long_hash] = short_hash
        short_hash_to_image[short_hash] = image
    return long_hash_to_short_hash[long_hash]

pretty = '<image_{short_hash}>'
ugly = r'<image_([a-zA-Z0-9]{1,10}?)>'
def img_go(image: Image.Image):
    return pretty.format(short_hash=_img_go(image))

def str_go(x: str) -> Optional[Image.Image]:
    try:
        b = base64.b64decode(x)
    except binascii.Error:
        b = x
    try:
        return img_go(Image.open(io.BytesIO(b)))
    except IOError:
        return x

def go(x: Any):
    if isinstance(x, Image.Image):
        return img_go(x)
    elif isinstance(x, str):
        return str_go(x)
    elif isinstance(x, bytes):
        return img_go(Image.open(io.BytesIO(x)))
    elif isinstance(x, dict):
        return {k: go(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return [go(v) for v in x]
    else:
        return x

def str_back(x: str) -> list[Union[str, Image.Image]]:
    parts = re.split(ugly, x)
    return [
        get_image_by_short_hash(x) if i % 2 == 1 else x.strip()
        for i, x in enumerate(parts) if x.strip()
    ]

def back(x):
    if isinstance(x, dict):
        return map(back, x.values())
    elif isinstance(x, Iterable):
        return map(back, x)
    elif isinstance(x, str):
        return str_back(x)
    else:
        return x


def call_openai(x: str):
    parts = re.split(ugly, x)
    content = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "text", "text": part})
        else:
            buffer = io.BytesIO()
            img = get_image_by_short_hash(part)
            if not hasattr(img, 'filename'):
                img.filename = ""
            img.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"}
            })
    
    return openai.chat.completions.create(
        model=model,
        messages=content,
    ).choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    image = Image.open(args.image_path)
    print(img_go(image))
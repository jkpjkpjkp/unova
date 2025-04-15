import shelve
from PIL import Image
import hashlib
from multiprocessing import Lock
from typing import Any, Iterable, Optional, Union, List, Literal
import base64
import binascii
import io
import re
import openai
import os
import argparse
import asyncio
import json
import fcntl
from tqdm import tqdm
import subprocess

_short_hash_to_image = shelve.open('image.shelve')
long_hash_to_short_hash = shelve.open('long_to_short.shelve')


def build_long_hash_from_short_hash():
    for short_hash, image in tqdm(_short_hash_to_image.items()):
        long_hash = hashlib.sha256(image.tobytes()).hexdigest()
        long_hash_to_short_hash[long_hash] = short_hash

def get_image_by_short_hash(short_hash):
    ret = _short_hash_to_image[short_hash]
    if not hasattr(ret, 'filename'):
        ret.filename = ""
    return ret.convert('RGB')

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
    image = image.convert('RGB')
    long_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if long_hash in long_hash_to_short_hash and long_hash_to_short_hash[long_hash]:
        return long_hash_to_short_hash[long_hash]
    with fcntl.flock(fcntl.open('.image_count', fcntl.O_RDWR), fcntl.LOCK_EX):
        with open('.image_count', 'r+') as f:
            tot = int(f.read()) + 1
            f.seek(0)
            f.write(str(tot))
    short_hash = int_to_0aA(tot)
    long_hash_to_short_hash[long_hash] = short_hash
    _short_hash_to_image[short_hash] = image
    return short_hash

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

def is_port_occupied(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


async def callopenai(x: str, tools: list[Literal['crop']]=['crop']):
    print(x)
    if not is_port_occupied(7912):
        subprocess.Popen(['cd', '~/Meta', '&&', 'uv', 'run', 'core/router_qwen.py'])
    parts = re.split(ugly, x)
    image_set = []
    content = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "text", "text": part})
        else:
            buffer = io.BytesIO()
            img = get_image_by_short_hash(part)
            if part in image_set:
                content.append({"type": "text", "text": f"<image_{part}>"})
                continue
            image_set.append(part)
            print(part)

            max_dim = 2000
            if max(img.width, img.height) > max_dim:
                scale = max_dim / max(img.width, img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"}
            })
            content.append({"type": "text", "text": f"<image_{part}>"})
    
    def retrieve(image_id):
        idc = re.match("image_(\d+)", image_id)
        if idc:
            image_id = int(idc.group(1)) - 1
            return get_image_by_short_hash(image_set[image_id])
        hs = re.match(ugly, image_id)
        if hs:
            return get_image_by_short_hash(hs.group(1))
        try:
            image_id = int(image_id)
            return image_set[image_id]
        except:
            pass
        return image_set[0]
    
    def crop(image_id=0, x1=0, y1=0, x2=1000, y2=1000):
        image = retrieve(image_id)
        image_dims = image.size
        x1 = x1 / 1000 * image_dims[0]
        y1 = y1 / 1000 * image_dims[1]
        x2 = x2 / 1000 * image_dims[0]
        y2 = y2 / 1000 * image_dims[1]
        return image.crop((x1, y1, x2, y2))
    
    tools = []
    if image_set and 'crop' in tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "crop",
                    "description": "Crop an image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string", "description": f"use 'image_1' to refer to the 1st image, or its representation, '<image_{image_set[0]}>'"},
                            "x1": {"type": "number", "description": "coordinates are from 0 to 1000"},
                            "y1": {"type": "number"},
                            "x2": {"type": "number"},
                            "y2": {"type": "number", "description": "coordinates are from 0 to 1000"},
                        },
                        "required": ["x1", "y1", "x2", "y2"]
                    }
                }
            }
        )

    messages=[{
        'role': 'user',
        'content': content
    }]

    response = openai.chat.completions.create(
        model='dummy',
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    while response.choices[0].message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": [tool_call]
        })
        tool_call = response.choices[0].message.tool_calls[0]
        func_name = tool_call.function.name
        if func_name == "crop":
            args = json.loads(tool_call.function.arguments)
            result = crop(**args)
            repr = img_go(result)
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            messages.append({
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": repr
                    }
                ],
                "tool_call_id": tool_call.id
            })
        else:
            raise ValueError(f"Unknown tool: {func_name}")

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    
    return response.choices[0].message.content

def cli():
    while True:
        short_hash = input("Enter short hash or image path (or press Enter/q to quit): ")
        try:
            if not short_hash:
                break
            if os.path.exists(short_hash):
                image = Image.open(short_hash)
                print(img_go(image))
            else:
                image = get_image_by_short_hash(short_hash)
                image.show()
        except KeyError:
            print(f"Error: Short hash '{short_hash}' not found.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    cli()
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
from gradio_client import Client, handle_file
import numpy as np
import tempfile

_short_hash_to_image = shelve.open('image.shelve')
long_hash_to_short_hash = shelve.open('long_to_short.shelve')
migra_db = shelve.open('migra_db.shelve')

def print_all_long_hash_to_short_hash():
    for key, value in long_hash_to_short_hash.items():
        print(f"{key}: {value}")


def build_long_hash_from_short_hash():
    for short_hash, image in tqdm(_short_hash_to_image.items()):
        long_hash = hashlib.sha256(image.tobytes()).hexdigest()
        long_hash_to_short_hash[long_hash] = short_hash

def get_image_by_short_hash(short_hash):
    try:
        ret = _short_hash_to_image[short_hash]
        if not hasattr(ret, 'filename'):
            ret.filename = ""
        return ret.convert('RGB')
    except:
        ret = migra_db[short_hash]
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
        short_hash = long_hash_to_short_hash[long_hash]
        try:
            assert _short_hash_to_image[short_hash] == image
        except:
            migra_db[short_hash] = image
            return short_hash
        return short_hash
    # with fcntl.flock(fcntl.open('.image_count', fcntl.O_RDWR), fcntl.LOCK_EX):
    #     with open('.image_count', 'r+') as f:
    #         tot = int(f.read()) + 1
    #         f.seek(0)
    #         f.write(str(tot))
    # short_hash = int_to_0aA(tot)
    # long_hash_to_short_hash[long_hash] = short_hash
    # _short_hash_to_image[short_hash] = image
    # return short_hash
    print("No short hash found")

def go_all(folder: str):
    for file in os.listdir(folder):
        if file.endswith('.png'):
            image = Image.open(os.path.join(folder, file))
            _img_go(image)

if __name__ == "__main__":
    go_all('/mnt/home/jkp/hack/diane/data/zerobench_images/zerobench')
    exit()

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

async def extract_image(x: str):
    return [part for part in str_back(x) if isinstance(part, Image.Image)]


def depth_estimator(image):
    client = Client("http://localhost:7860/")
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name
        try:
            result_path = client.predict(
                    image=handle_file(image_path),
                    api_name="/predict"
            )
        finally:
            os.remove(image_path)

    depth_image = Image.open(result_path).convert('L')
    depth_array = np.array(depth_image)

    return depth_array

def sam2(image):
    client = Client("http://localhost:7861/")
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name
    try:
        result = client.predict(
                input_image=handle_file(image_path),
                api_name="/predict"
        )
    finally:
        os.remove(image_path)
    return np.array([np.array(Image.open(x['image'])) for x in result])


async def callopenai(x: str, model='gemini-2.0-flash',tools: list[Literal['crop']]=[]):
    print(x)
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
        idc = re.match(r"image_(\d+)", image_id)
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
        model=model,
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
def check_and_repair_shelf():
    corrupted_keys = []
    print("Checking shelf for corrupted entries...")
    for short_hash in list(_short_hash_to_image.keys()):
        try:
            # Attempt to load the image
            image = _short_hash_to_image[short_hash]
            # Verify it’s a valid PIL Image by accessing a property
            image.tobytes()
            # Update long_hash mapping
            long_hash = hashlib.sha256(image.tobytes()).hexdigest()
            long_hash_to_short_hash[long_hash] = short_hash
        except Exception as e:
            print(f"Corrupted key '{short_hash}': {e}")
            corrupted_keys.append(short_hash)
# import dbm
# import pickle
# import shelve

# try:
#     db = dbm.open('image.shelve', 'r')
# except dbm.error as e:
#     print(f"Error opening database: {e}")
#     exit(1)

# new_shelf = shelve.open('recovered_data.db', 'c')

# key = db.firstkey()
# while key is not None:
#     try:
#         value = pickle.loads(db[key])
#         new_shelf[key.decode()] = value
#     except Exception as e:
#         print(f"Error processing key {key}: {e}")
#     key = db.nextkey(key)

# new_shelf.close()
# db.close()
# # Run the repair
# if __name__ == "__main__":
#     # check_and_repair_shelf()
#     cli()

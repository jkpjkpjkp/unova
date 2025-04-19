from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select, delete
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import List, Optional, Tuple, Dict, Any, Self, Iterator
import hashlib
import os
from PIL import Image
import functools
import numpy as np
import hashlib
from typing import Any, Iterable, Optional, Union, Literal
import base64
import binascii
import io
import re
import openai
import os
import json
from tqdm import tqdm
from gradio_client import Client, handle_file
import numpy as np
import tempfile
import contextlib
import fcntl

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
        return Self(self._img.crop(xyxy or self.bbox))
    def crop1000(self, box: tuple):
        x, y = self.shape
        return self.crop(box[0] / 1000 * x, box[1] / 1000 * y, box[2] / 1000 * x, box[3] / 1000 * y)
    
    @property
    def image(self):
        return self._img if isinstance(self._img, Image.Image) else functools.reduce(lambda a, b: a.alpha_composite(b), self._img, initial=Image.new('RGBA', self._img[0].size, (0, 0, 0, 0)))
    @property
    def bbox(self):
        return self.image.getbbox()
    def present(self, mode='raw') -> list[Image.Image]:
        if mode == 'raw':
            return [self.crop(self.image)]
        elif mode == 'box':
            return [self.crop(self.image).to('RGB')]
        elif mode == 'cascade':
            center = tuple(int, int)(self.center())
            x, y = self.bbox // 2
            return [self.crop(xyxy=(center[0]-x*2**i, center[1]-y*2**i, center[0]+x*2**i, center[1]+y*2**i)) for i in range(3)]

VE = VisualEntity
db_name = "main.db"
tag='zerobench'


class MyHash:
    @property
    def hash(self):
        code = '\n'.join(str(getattr(self, field)) for field in self._hash_fields)
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id
    
    def __hash__(self):
        return int.from_bytes(self.id or self.hash, 'big')
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class ImageHashToID(SQLModel, table=True):
    id: bytes = Field(primary_key=True) # long hash
    short_hash: str


class Graph(MyHash, SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph: str
    prompt: str
    tags: list[str] = Field(sa_column=Column(JSON))

    runs: list["Run"] = Relationship(back_populates="graph")
    _hash_fields = ('graph', 'prompt')
    
    @property
    def experience(self):
        return [(x.modification, x.graph.score) for x in get(Ron, Graph, tag=tag)[self]]
    
    @property
    def score(self):
        rund = get(Run, Graph, Task)[self]
        if not rund:
            return 0.5
        return (sum(sum(run.correct for run in runs) / len(runs) for runs in rund.values()) + 1) / (len(rund) + 2)


class Task(MyHash, SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    task: str
    answer: str
    tags: list[str] = Field(sa_column=Column(JSON))

    runs: list["Run"] = Relationship(back_populates="task")
    _hash_fields = ('task', 'answer')


class RonRunLink(SQLModel, table=True):
    ron_id: Optional[bytes] = Field(default=None, foreign_key="ron.id", primary_key=True)
    run_id: Optional[bytes] = Field(default=None, foreign_key="run.id", primary_key=True)


class Run(MyHash, SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph_id: bytes = Field(foreign_key="graph.id")
    task_id: bytes = Field(foreign_key="task.id")
    log: Dict[str, Any] = Field(sa_column=Column(JSON))
    final_output: str | None = Field(default=None)
    correct: bool
    tags: list[str] = Field(sa_column=Column(JSON))

    graph: Graph = Relationship(back_populates="runs")
    task: Task = Relationship(back_populates="runs")
    used_in: List["Ron"] = Relationship(back_populates="runs", link_model=RonRunLink)
    _hash_fields = ('graph_id', 'task_id', 'log', 'tags')


class Groph(MyHash, SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph: str
    prompt: str
    tags: list[str] = Field(sa_column=Column(JSON))

    rons: list["Ron"] = Relationship(back_populates="groph")
    _hash_fields = ('graph', 'prompt')


class Ron(MyHash, SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    groph_id: bytes = Field(foreign_key='groph.id')
    runs: List["Run"] = Relationship(back_populates="used_in", link_model=RonRunLink)
    log: Dict[str, Any] = Field(sa_column=Column(JSON))
    final_output: bytes = Field(foreign_key='graph.id')
    tags: list[str] = Field(sa_column=Column(JSON))

    groph: Groph = Relationship(back_populates='rons')
    _hash_fields = ('groph_id', 'log', 'tags')

    @property
    def graph(self):
        return self.runs[0].graph

    @property
    def new_graph(self):
        return get_by_id(Graph, self.new_graph_id)
    
    @property
    def modification(self):
        return self.log['modification']


_engine = create_engine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(_engine)


def test_ron_has_runid():
    with Session(_engine) as session:
        ron = session.exec(select(Ron)).first()
        assert ron.id
        assert len(ron.runs)


def DANGER_DANGER_DANGER_remove_dangling():  # everything with illegal foreign key (no related main key) is deleted
    with Session(_engine) as session:
        session.exec(delete(RonRunLink).where(
            ~RonRunLink.ron_id.in_(select(Ron.id)) |
            ~RonRunLink.run_id.in_(select(Run.id))
        ))
        session.exec(delete(Run).where(
            ~Run.graph_id.in_(select(Graph.id)) |
            ~Run.task_id.in_(select(Task.id))
        ))
        session.exec(delete(Ron).where(
            ~Ron.groph_id.in_(select(Groph.id)) |
            ~Ron.final_output.in_(select(Graph.id))
        ))
        session.commit()


def go(x):
    x.id = x.id or x.hash
    with Session(_engine) as session:
        merged_x = session.merge(x)
        session.commit()
        session.refresh(merged_x)
    return merged_x

def get_graph_from_a_folder(folder: str, groph: bool = False):
    with open(os.path.join(folder, "graph.py"), "r") as f:
        graph = f.read()
    with open(os.path.join(folder, "prompt.py"), "r") as f:
        prompt = f.read()
    graph = (Graph if not groph else Groph)(graph=graph, prompt=prompt)
    return go(graph)

def get_by_id(ret_type, id: bytes):
    with Session(_engine) as session:
        return session.exec(select(ret_type).where(ret_type.id == id)).first()

def read_tasks_from_a_parquet(filepath: str | list[str], tag: Optional[str] = None, keys: Tuple[str, str, str] = ('question_text', 'question_answer', 'question_images_decoded'), tag_key: Optional[str] = None):
    import polars as pl
    from tqdm import tqdm
    df = pl.read_parquet(filepath)
    for row in tqdm(df.iter_rows(named=True)):
        images = row[keys[2]]
        images = [x['bytes'] for x in images] if isinstance(images, list) else [images['bytes']]
        images = img_go(images)
        if isinstance(images, list):
            images = ' '.join(images)
        tags = []
        if tag:
            tags.append(tag)
        if tag_key:
            tags.append(row[tag_key])
        go(Task(task=images + ' ' + row[keys[0]], answer=str(row[keys[1]]), tags=tags))
def recover_image_from_a_parquet(filepath: str | list[str], tag: Optional[str] = None, keys: Tuple[str, str, str] = ('question_text', 'question_answer', 'question_images_decoded'), tag_key: Optional[str] = None):
    import polars as pl
    from tqdm import tqdm
    df = pl.read_parquet(filepath)
    for row in tqdm(df.iter_rows(named=True)):
        images = row[keys[2]]
        images = [x['bytes'] for x in images] if isinstance(images, list) else [images['bytes']]
        images = img_go(images)


def test_get_graph_from_a_folder():
    with Session(_engine) as session:
        print(len(session.exec(select(Graph)).all()))
    get_graph_from_a_folder("sample/cot")
    with Session(_engine) as session:
        print(len(session.exec(select(Graph)).all()))
def test_get_groph_from_a_folder():
    with Session(_engine) as session:
        print(len(session.exec(select(Groph)).all()))
    get_graph_from_a_folder("sampo/bflow", groph=True)
    with Session(_engine) as session:
        print(len(session.exec(select(Groph)).all()))
def test_read_tasks_from_a_parquet():
    with Session(_engine) as session:
        print(len(session.exec(select(Task)).all()))
    read_tasks_from_a_parquet("/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet", tag='zerobench')
    with Session(_engine) as session:
        print(len(session.exec(select(Task)).all()))


def get(*args, tag=None):
    n = len(args)
    with Session(_engine) as session:
        if tag:
            aaa = session.exec(select(args[0]).where(~args[0].tags.contains('del'))).where(args[0].tags.contains(tag)).all()
        else:
            aaa = session.exec(select(args[0]).where(~args[0].tags.contains('del'))).all()
        if n == 1:
            return aaa
        group1 = session.exec(select(args[1])).all()
        if n == 2:
            ret = {g1: [] for g1 in group1}
            for r in aaa:
                ret[getattr(r, args[1].__name__.lower())].append(r)
            return ret
        if n == 3:
            ret = {g1: {} for g1 in group1}
            for r in aaa:
                k1 = getattr(r, args[1].__name__.lower())
                k2 = getattr(r, args[2].__name__.lower())
                if k2 not in ret[k1]:
                    ret[k1][k2] = []
                ret[k1][k2].append(r)
            return ret
        raise ValueError(f"Invalid number of arguments: {n}")



async def test_get(tag=None):
    best_graph = get_strongest_graph()
    task_stat = get_task_stat(tag=tag)
    
    print(get(Run, Graph, tag=tag)[best_graph])
    runs_for_best_graph = get(Run, Graph, tag=tag)[best_graph]

def where(x, *args, tag=None):
    print('HERE')
    n = len(args)
    assert n == 1
    with Session(_engine) as session:
        aaa = session.exec(select(args[0])).all()
        ret = []
        for r in aaa:
            print(getattr(r, type(x).__name__.lower()).id)
            if getattr(r, type(x).__name__.lower()) == x:
                ret.append(r)
        return ret


def remove(x):
    with Session(_engine) as session:
        session.delete(x)
        session.commit()

def count_rows(table):
    with Session(_engine) as session:
        return session.exec(select(func.count()).select_from(table)).first()

def test_get():
    print(get(Run, Graph, tag=tag))
def test_count_rows():
    print(count_rows(Run))

def get_strongest_graph():
    with Session(_engine) as session:
        task_avg_sq = (
            select(
                Run.graph_id,
                Run.task_id,
                func.avg(Run.correct).label("task_avg"),
            )
            .join(Task, Run.task_id == Task.id)
            .where(Task.tags.contains('zerobench'))
            .group_by(Run.graph_id, Run.task_id)
            .subquery("task_averages")
        )
        graph_avg_sq = (
            select(
                task_avg_sq.c.graph_id,
                func.avg(task_avg_sq.c.task_avg).label("overall_avg"),
            )
            .group_by(task_avg_sq.c.graph_id)
            .subquery("graph_averages")
        )
        stmt = (
            select(Graph, graph_avg_sq.c.overall_avg)
            .join(graph_avg_sq, Graph.id == graph_avg_sq.c.graph_id)
            .order_by(graph_avg_sq.c.overall_avg.desc())
        )
        
        result = session.exec(stmt).all()
        print(result)
        exit()
        if result:
            strongest_graph, score = result
            print(f"Strongest graph found: ID={strongest_graph.id}, Score={score}")
            return strongest_graph
        else:
            raise ValueError("No runs found to determine the strongest graph.")

def find_hardest_tasks(top_n: int = 10, tag=None):
    with Session(_engine) as session:
        stmt = (
            select(Task, func.avg(Run.correct).label("avg_correctness"))
            .join(Run, Task.id == Run.task_id)
            .group_by(Task.id)
            .order_by(func.avg(Run.correct).asc()) # Ascending order for lowest correctness first
            .limit(top_n)
        )
        results = session.exec(stmt).all()
        if results:
            print(f"Top {top_n} hardest tasks:")
            for task, avg_correctness in results:
                print(f"  Task ID: {task.id}, Avg Correctness: {avg_correctness:.4f}")
            return [x[0] for x in results]
        else:
            print("No runs found to determine hardest tasks.")
            return []

def test_get_strongest_graph():
    assert isinstance(get_strongest_graph(), Graph)
def test_find_hardest_tasks():
    ret = find_hardest_tasks(2)
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert isinstance(ret[0], Task)

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


# Define a context manager for file locking
@contextlib.contextmanager
def flock_fd(fd):
    fcntl.flock(fd, fcntl.LOCK_EX)  # Acquire an exclusive lock
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)  # Release the lock

# Ensure the file exists with an initial value of 0
if not os.path.exists('.image_count'):
    with open('.image_count', 'w') as f:
        f.write('0')

# Open the file, lock it, and increment the counter
with open('.image_count', 'r+') as f:
    with flock_fd(f.fileno()):
        tot = int(f.read()) + 1
        f.seek(0)
        f.write(str(tot))
        f.truncate()

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
    with open('.image_count', 'r+') as f:
        with flock_fd(f.fileno()):
            tot = int(f.read()) + 1
            f.seek(0)
            f.write(str(tot))
            f.truncate()
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
    if isinstance(x, VE):

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
    assert isinstance(image, Image.Image)
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
            image = _short_hash_to_image[short_hash]
            image.tobytes()
            long_hash = hashlib.sha256(image.tobytes()).hexdigest()
            long_hash_to_short_hash[long_hash] = short_hash
        except Exception as e:
            print(f"Corrupted key '{short_hash}': {e}")
            corrupted_keys.append(short_hash)

def all_tests():
    test_get_graph_from_a_folder()
    test_get_groph_from_a_folder()
    # test_read_tasks_from_a_parquet()
    test_get()
    test_count_rows()
    test_get_strongest_graph()
    test_find_hardest_tasks()


if __name__ == "__main__":
    all_tests()
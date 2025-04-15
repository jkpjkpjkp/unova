from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select, delete
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import List, Optional, Tuple, Type, Dict, Any
import hashlib
import os
from image_shelve import go as img_go
import json
db_name = "main.db"
tag='zerobench'

class MyHash:
    @property
    def hash(self):
        code = '\n'.join(str(getattr(self, field)) for field in self._hash_fields)
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Graph(MyHash, SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph: str
    prompt: str
    tags: list[str] = Field(sa_column=Column(JSON))

    runs: list["Run"] = Relationship(back_populates="graph")
    _hash_fields = ('graph', 'prompt')
    
    @property
    def experience(self):
        return [(x.modification, x.graph.score) for x in get(Ron, Graph, tag=tag)[self.id]]
    
    @property
    def score(self):
        rund = gett(Run, Graph, Task)[self.id]
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
    rons: List["Ron"] = Relationship(back_populates="runs", link_model=RonRunLink)
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
    run_ids: list[bytes] = Field(foreign_key='run.id', sa_column=Column(JSON))
    log: Dict[str, Any] = Field(sa_column=Column(JSON))
    final_output: bytes = Field(foreign_key='graph.id')
    tags: list[str] = Field(sa_column=Column(JSON))

    groph: Groph = Relationship(back_populates='rons')
    runs: List["Run"] = Relationship(back_populates="rons", link_model=RonRunLink)
    _hash_fields = ('groph_id', 'run_ids', 'log', 'tags')

    
    @property
    def graphs(self):
        ret = set()
        for run in self.runs:
            ret.add(run.graph)
        return ret

    @property
    def new_graph(self):
        return get_by_id(Graph, self.new_graph_id)
    
    @property
    def modification(self):
        return self.log['modification']

engine = create_engine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(engine)


def go(x):
    x.id = x.id or x.hash
    with Session(engine) as session:
        merged_x = session.merge(x)
        session.commit()
        session.refresh(merged_x)
    return merged_x

def get_graph_from_a_folder(folder: str, groph: bool = False):
    with open(os.path.join(folder, "graph.py"), "r") as f:
        graph = f.read()
    with open(os.path.join(folder, "prompt.py"), "r") as f:
        prompt = f.read()
    graph = (Graph if not groph else Groph)(graph=graph, prompt=prompt, task_tag='counting')
    return go(graph)

def get_by_id(ret_type, id: bytes):
    with Session(engine) as session:
        return session.exec(select(ret_type).where(ret_type.id == id)).first()
def graph(graph_id):
    return get_by_id(Graph, graph_id)
def task(task_id):
    return get_by_id(Task, task_id)

def test_get_graph_from_a_folder():
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))
    get_graph_from_a_folder("sample/cot")
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))

def test_read_groph_from_a_folder():
    with Session(engine) as session:
        print(len(session.exec(select(Groph)).all()))
    get_graph_from_a_folder("sampo/bflow", groph=True)
    with Session(engine) as session:
        print(len(session.exec(select(Groph)).all()))

def read_tasks_from_a_parquet(filepath: str | list[str], tag: Optional[str] = None, keys: Tuple[str, str, str] = ('question_text', 'question_answer', 'question_images_decoded'), tag_key: Optional[str] = None):
    import polars as pl
    from tqdm import tqdm
    from loguru import logger
    df = pl.read_parquet(filepath)
    for row in tqdm(df.iter_rows(named=True)):
        images = row[keys[2]]
        images = [x['bytes'] for x in images] if isinstance(images, list) else [images['bytes']]
        images = img_go(images)
        if isinstance(images, list):
            images = ' '.join(images)
        try:
            task = Task(task=images + ' ' + row[keys[0]], answer=float(row[keys[1]]))
        except ValueError as e:
            logger.info(f"None float answer: {row[keys[1]]}")
            task = Task(task=images + ' ' + row[keys[0]], answer=str(row[keys[1]]))
        task.tags = []
        if tag:
            task.tags.append(tag)
        if tag_key:
            task.tags.append(row[tag_key])
        go(task)

def test_read_tasks_from_a_parquet():
    with Session(engine) as session:
        print(len(session.exec(select(Task)).all()))
    read_tasks_from_a_parquet("/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet")
    with Session(engine) as session:
        print(len(session.exec(select(Task)).all()))

def print_graph_stat(folder: str):
    graph = get_graph_from_a_folder(folder)
    with Session(engine) as session:
        runs = session.exec(select(Run).where(Run.graph == graph)).all()
        print(sum(run.correct for run in runs))
        print(len(runs))
        tru = {}
        for run in runs:
            question = run.task.task.lower()
            if not run.task_id in tru:
                tru[run.task_id] = (0, 0)
            tru[run.task_id] = (tru[run.task_id][0] + run.correct, tru[run.task_id][1] + 1)
        print(sum(x[0] / x[1] for x in tru.values()))
        print(sum(int(bool(x[0])) for x in tru.values()))
        print(len(tru))
        for run in runs:
            print(run.task.answer)
            print(run.log['__ANSWER__'] if '__ANSWER__' in run.log else 'N/A')

def DANGER_DANGER_DANGER_test_add_tag_to_task():
    with Session(engine) as session:
        tasks = session.exec(select(Task)).all()
        for task in tasks:
            task.tags = ['zerobench']
            session.merge(task)
        session.commit()

def get(ret_type, group_by, tag=None):
    with Session(engine) as session:
        if tag:
            aaa = session.exec(select(ret_type).where(ret_type.tags.contains(tag))).all()
        else:
            aaa = session.exec(select(ret_type)).all()
        group = session.exec(select(group_by)).all()
        ret = {g.id: [] for g in group}
        for r in aaa:
            print(type(r))
            ret[getattr(r, group_by.__name__.lower()).id].append(r)
    return ret

def gett(ret_type, group_by1, group_by2, tag=None):
    # TODO: add tag
    with Session(engine) as session:
        aaa = session.exec(select(ret_type)).all()
        group1 = session.exec(select(group_by1)).all()
        ret = {g1.id: {} for g1 in group1}
        for r in aaa:
            id1 = getattr(r, group_by1.__name__.lower()).id
            id2 = getattr(r, group_by2.__name__.lower()).id
            if not id2 in ret[id1]:
                ret[id1][id2] = []
            ret[id1][id2].append(r)
    return ret

def count_rows(query_type):
    with Session(engine) as session:
        count_statement = select(func.count()).select_from(query_type)
        return session.exec(count_statement).first()

def test_get():
    print(get(Run, Graph, tag=tag))

def test_len():
    print(count_rows(Run))

if __name__ == "__main__":
    # print_graph_stat("/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7")
    # read_tasks_from_a_parquet(["/home/jkp/Téléchargements/mmiq-00000-of-00001.parquet"], tag='mmiq', keys=('question_en', 'answer', 'image'))

    # get_graph_from_a_folder("sampo/bflow", groph=True)
    test_get()
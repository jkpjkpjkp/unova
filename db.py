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
    
    def __hash__(self):
        return int.from_bytes(self.id or self.hash, 'big')
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


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
        print(ron.id)
        print(ron.runs)

if __name__ == "__main__":
    test_ron_has_runid()
    exit()

def DANGER_DANGER_DANGER_trim():  # everything with illegal foreign key (no related main key) is deleted
    with Session(_engine) as session:
        stmt = delete(RonRunLink).where(
            ~RonRunLink.ron_id.in_(select(Ron.id)) |
            ~RonRunLink.run_id.in_(select(Run.id))
        )
        session.execute(stmt)

        stmt = delete(Run).where(
            ~Run.graph_id.in_(select(Graph.id)) |
            ~Run.task_id.in_(select(Task.id))
        )
        session.execute(stmt)

        stmt = delete(Ron).where(
            ~Ron.groph_id.in_(select(Groph.id)) |
            ~Ron.final_output.in_(select(Graph.id))
        )
        session.execute(stmt)
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
    from loguru import logger
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

def print_graph_stat(folder: str):
    graph = get_graph_from_a_folder(folder)
    with Session(_engine) as session:
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
    with Session(_engine) as session:
        tasks = session.exec(select(Task)).all()
        for task in tasks:
            task.tags = ['zerobench']
            session.merge(task)
        session.commit()

def get(*args, tag=None):
    n = len(args)
    with Session(_engine) as session:
        if tag:
            aaa = session.exec(select(args[0]).where(args[0].tags.contains(tag))).all()
        else:
            aaa = session.exec(select(args[0])).all()
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
                if not k2 in ret[k1]:
                    ret[k1][k2] = []
                ret[k1][k2].append(r)
            return ret
        raise ValueError(f"Invalid number of arguments: {n}")

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

def all_tests():
    test_get_graph_from_a_folder()
    test_get_groph_from_a_folder()
    test_read_tasks_from_a_parquet()
    test_get()
    test_count_rows()

if __name__ == "__main__":
    read_tasks_from_a_parquet("/home/jkp/Téléchargements/mmiq-00000-of-00001.parquet", tag='mmiq', tag_key='category', keys=('question_en', 'answer', 'image'))
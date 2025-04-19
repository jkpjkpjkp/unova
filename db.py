from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any
import hashlib
import os
import sys
import functools
from action_node import operators

db_name = "runs.db"

class MyHash:
    def __hash__(self):
        def hash():
            code = '\n'.join(str(getattr(self, field)) for field in self._hash_fields)
            self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id or int.from_bytes(hash(), 'big')
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class Graph(MyHash, SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph: str
    prompt: str
    father: 'Graph' | None = Relationship(backpopulates="children")
    change: str | None = Field(default=None)
    children: list['Graph'] = Relationship(backpopulates="father")

    runs: list["Run"] = Relationship(back_populates="graph")
    _hash_fields = ('graph', 'prompt')

    @classmethod
    def read(foldername):
        with open(os.path.join(foldername, "graph.py"), "r") as f:
            graph = f.read()
        with open(os.path.join(foldername, "prompt.py"), "r") as f:
            prompt = f.read()
        return Graph(graph=graph, prompt=prompt)

    @property
    def run(self):
        graph_code = self.graph + '\n' + self.prompt
        namespace = {'__name__': '__exec__', '__package__': None}
        
        try:
            exec(graph_code, namespace)
            graph = namespace.get("Graph")(operators=operators, prompt_custom=namespace)
        except Exception:
            print("--- Error reading graph code ---")
            print(graph_code)
            print("--- error ---")
            raise

        def extract_local_variables_wrapper(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                original = sys.gettrace()
                captured_locals = {}
                def trace(frame, event, _arg):
                    if event == 'return' and frame.f_code is func.__code__:
                        nonlocal captured_locals
                        captured_locals = frame.f_locals
                    return trace
                sys.settrace(trace)
                try:
                    result = await func(*args, **kwargs)
                except:
                    sys.settrace(original)
                    raise
                sys.settrace(original)
                captured_locals = dict(captured_locals)
                captured_locals.pop('self')
                return result, captured_locals
            return wrapper
        
        def log_to_db_wrapper(graph_id): 
            def decorator(func):
                @functools.wraps(func)
                async def wrapper(task):
                    answer = task.pop('question_answer')
                    result, captured_locals = await func(task)
                    task['question_id'] = answer
                    task_id = task['question_id']
                    correct = (result == task['question_answer'])
                    
                    run = Run(
                        graph_id=graph_id,
                        task_id=task_id,
                        log=captured_locals,
                        final_output=result,
                        correct=correct
                    )
                    
                    with Session(_engine) as session:
                        session.add(run)
                        session.commit()
                    
                    return result
                return wrapper
            return decorator
        
        return extract_local_variables_wrapper(graph.run)


class Run(MyHash, SQLModel, table=True):
    graph_id: int = Field(primary_key=True, foreign_key="graph.id")
    task_id: str = Field(primary_key=True)
    log: Dict[str, Any] = Field(sa_column=Column(JSON))
    final_output: str | None = Field(default=None)
    correct: bool

    graph: Graph = Relationship(back_populates="runs")
    _hash_fields = ('graph_id', 'task_id', 'log')

_engine = create_engine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(_engine)

import polars as pl
from PIL import Image
import io

df = pl.read_parquet('/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet')
tasks = df['question_id'].to_list()

def get_high_variation_task(k=1):
    ret = []
    with Session(_engine) as session:
        run_task_ids = session.exec(select(Run.task_id)).all()
        for task_id in tasks:
            if task_id not in run_task_ids:
                ret.append(task_id)
    if len(ret) >= k:
        return ret[:k]
    with Session(_engine) as session:
        ret.extend(session.exec(select(Run.task_id).group_by(Run.task_id).order_by(func.std(Run.correct).desc()).limit(k - len(ret))).all())
    return ret

def put(x):
    if isinstance(x, Graph):
        x.id = x.id or x.__hash__()
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
    graph = Graph(graph=graph, prompt=prompt)
    return put(graph)

def test_get_graph_from_a_folder():
    with Session(_engine) as session:
        print(len(session.exec(select(Graph)).all()))
    get_graph_from_a_folder("sample/cot")
    with Session(_engine) as session:
        print(len(session.exec(select(Graph)).all()))

def get_strongest_graph(k=1):
    with Session(_engine) as session:
        return session.exec(select(Graph).order_by(func.avg(Run.correct).desc()).limit(k)).all()
    
def get_hardest_task(k=1):
    with Session(_engine) as session:
        return session.exec(
            select(Run.task_id)
            .group_by(Run.task_id)
            .order_by(func.avg(Run.correct).asc())
            .limit(k)
        ).all()
    
def test_get_strongest_graph():
    assert isinstance(get_strongest_graph(), Graph)
def test_get_hardest_task():
    ret = get_hardest_task(2)
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert isinstance(ret[0], str)
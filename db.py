from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any
import hashlib
import os
from PIL import Image
import numpy as np
from gradio_client import Client, handle_file
import tempfile

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

    runs: list["Run"] = Relationship(back_populates="graph")
    _hash_fields = ('graph', 'prompt')


class Run(MyHash, SQLModel, table=True):
    graph_id: int = Field(primary_key=True, foreign_key="graph.id")
    task_id: str = Field(primary_key=True)
    log: Dict[str, Any] = Field(sa_column=Column(JSON))
    final_output: str | None = Field(default=None)
    correct: bool

    graph: Graph = Relationship(back_populates="runs")
    _hash_fields = ('graph_id', 'task_id', 'log', 'tags')


_engine = create_engine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(_engine)


def put(x):
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

def get_by_id(ret_type, id: bytes):
    with Session(_engine) as session:
        return session.exec(select(ret_type).where(ret_type.id == id)).first()


def get_strongest_graph(k=1):
    with Session(_engine) as session:
        task_avg_sq = (
            select(
                Run.graph_id,
                Run.task_id,
                func.avg(Run.correct).label("task_avg"),
            )
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

def get_hardest_task(k=1):
    TODO
    
def test_get_strongest_graph():
    assert isinstance(get_strongest_graph(), Graph)
def test_get_hardest_task():
    ret = get_hardest_task(2)
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert isinstance(ret[0], str)


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





from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select, delete
from sqlalchemy import Column
from sqlalchemy.types import JSON
from typing import List, Optional, Tuple
import hashlib
import os
from image_shelve import go as img_go, put_log, get_log
db_name = "main.db"

class Graph(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph: str
    prompt: str
    task_tag: str
    runs: list["Run"] = Relationship(back_populates="graph")

    @property
    def hash(self):
        self.graph = self.graph.strip(' \n')
        self.prompt = self.prompt.strip(' \n')
        code = self.graph + '\n' + self.prompt + '\n' + self.task_tag
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Task(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    task: str
    answer: float
    tags: list[str] = Field(sa_column=Column(JSON))
    runs: list["Run"] = Relationship(back_populates="task")

    @property
    def hash(self):
        code = self.task + f"\\boxed{{{self.answer}}}"
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class RonRunLink(SQLModel, table=True):
    ron_id: Optional[bytes] = Field(
        default=None, foreign_key="ron.id", primary_key=True
    )
    run_id: Optional[bytes] = Field(
        default=None, foreign_key="run.id", primary_key=True
    )

class Run(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph_id: bytes = Field(foreign_key="graph.id")
    task_id: bytes = Field(foreign_key="task.id")
    log_id: int
    correct: bool

    graph: Graph = Relationship(back_populates="runs")
    task: Task = Relationship(back_populates="runs")
    rons: List["Ron"] = Relationship(back_populates="runs", link_model=RonRunLink)

    @property
    def hash(self):
        code = str(self.graph_id) + '\n' + str(self.task_id) + '\n' + str(self.log_id) + '\n' + str(self.correct)
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Opti(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph: str
    prompt: str
    rons: list["Ron"] = Relationship(back_populates="opti")

    @property
    def hash(self):
        code = self.graph + '\n' + self.prompt
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Ron(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    opti_id: bytes = Field(foreign_key='opti.id')
    log_id: int
    new_graph_id: bytes = Field(foreign_key='graph.id')
    opti: Opti = Relationship(back_populates='rons')

    runs: List["Run"] = Relationship(back_populates="rons", link_model=RonRunLink)

    @property
    def hash(self):
        code = str(self.log_id) + '\n' + str(self.new_graph_id)
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id
    
    def graphs(self):
        ret = set()
        for run in self.runs:
            ret.add(run.graph)
        return ret


engine = create_engine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(engine)


def go(x):
    x.id = x.id or x.hash
    with Session(engine) as session:
        merged_x = session.merge(x)
        session.commit()
        session.refresh(merged_x)
    return merged_x

def read_graph_from_a_folder(folder: str):
    with open(os.path.join(folder, "graph.py"), "r") as f:
        graph = f.read()
    with open(os.path.join(folder, "prompt.py"), "r") as f:
        prompt = f.read()
    graph = Graph(graph=graph, prompt=prompt, task_tag='counting')
    return go(graph)

def read_opti_from_a_folder(folder: str):
    with open(os.path.join(folder, "graph.py"), "r") as f:
        graph = f.read()
    with open(os.path.join(folder, "prompt.py"), "r") as f:
        prompt = f.read()
    opti = Opti(graph=graph, prompt=prompt)
    return go(opti)

def DANGER_DANGER_DANGER_test_read_graph_from_a_folder():
    with Session(engine) as session:
        session.exec(delete(Graph))
        session.commit()
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))
    read_graph_from_a_folder("sample/cot")
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))

def test_read_opti_from_a_folder():
    with Session(engine) as session:
        print(len(session.exec(select(Opti)).all()))
    read_opti_from_a_folder("sampo/bflow")
    with Session(engine) as session:
        print(len(session.exec(select(Opti)).all()))

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

def check_7(folder: str):
    graph = read_graph_from_a_folder(folder)
    with Session(engine) as session:
        runs = session.exec(select(Run).where(Run.graph == graph)).all()
        print(sum(run.correct for run in runs))
        print(len(runs))
        tru = {}
        for run in runs:
            if not run.task_id in tru:
                tru[run.task_id] = (0, 0)
            tru[run.task_id] = (tru[run.task_id][0] + run.correct, tru[run.task_id][1] + 1)
        print(sum(x[0] / x[1] for x in tru.values()))
        print(len(tru))
        for run in runs:
            print(run.task.answer)
            print(get_log(run.log_id)['__ANSWER__'])

def add_tag_to_task():
    with Session(engine) as session:
        tasks = session.exec(select(Task)).all()
        for task in tasks:
            task.tags = ['zerobench']
            session.merge(task)
        session.commit()

def get(ret_type, group_by):
    with Session(engine) as session:
        ret = session.select(ret_type).all()
        group = session.select(group_by).all()
    ret = {g.id: [] for g in group}
    for r in ret:
        ret[getattr(r, str(group_by).lower()).id].append(r)
    return ret

def gett(ret_type, group_by1, group_by2):
    with Session(engine) as session:
        ret = session.select(ret_type).all()
        group1 = session.select(group_by1).all()
    ret = {g1.id: {} for g1 in group1}
    for r in ret:
        id1 = getattr(r, str(group_by1).lower()).id
        id2 = getattr(r, str(group_by2).lower()).id
        if not id2 in ret[id1]:
            ret[id1][id2] = []
        ret[id1][id2].append(r)
    return ret

if __name__ == "__main__":
    # check_7("/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7")
    read_tasks_from_a_parquet(["/home/jkp/Téléchargements/mmiq-00000-of-00001.parquet"], tag='mmiq', keys=('question_en', 'answer', 'image'))
from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select, delete
from sqlalchemy import Column
from sqlalchemy.types import JSON
from typing import List, Optional
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

def read_tasks_from_a_parquet(filepath: str):
    import polars as pl
    from tqdm import tqdm
    from loguru import logger
    df = pl.read_parquet(filepath)
    for row in tqdm(df.iter_rows(named=True)):
        images = [x['bytes'] for x in row["question_images_decoded"]]
        images = img_go(images)
        if isinstance(images, list):
            images = ' '.join(images)
        try:
            task = Task(task=images + ' ' + row["question_text"], answer=float(row["question_answer"]))
        except ValueError:
            logger.warning(f"Error parsing {row['question_text']} {row['question_answer']}")
            continue
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

if __name__ == "__main__":
    check_7("/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7")

from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select, delete
import hashlib
import os
from image_shelve import go as img_go, put_log
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
    tags: list[str]
    runs: list["Run"] = Relationship(back_populates="task")

    @property
    def hash(self):
        code = self.task + f"\\boxed{{{self.answer}}}"
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Run(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph_id: bytes = Field(foreign_key="graph.id")
    task_id: bytes = Field(foreign_key="task.id")
    log_id: int
    correct: bool

    graph: Graph = Relationship(back_populates="runs")
    task: Task = Relationship(back_populates="runs")

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

class Ron(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    run_ids: list[bytes] # Field(foreign_key="run.id")
    opti_id: bytes = Field(foreign_key='opti.id')
    log_id: int
    to: bytes = Field(foreign_key='graph.id')
    opti: Opti = Relationship(back_populates='rons')

    @property
    def hash(self):
        code = str(self.run_ids) + '\n' + str(self.log_id) + '\n' + str(self.to)
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id


engine = create_engine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(engine)

class RRun:
    graph: Graph
    task: Task
    log: dict
    correct: bool

def go(x):
    if isinstance(x, RRun):
        x = Run(
            graph_id=x.graph.hash,
            task_id=x.task.hash,
            log_id=put_log(x.log),
            correct=x.correct,
        )
    x.id = x.id or x.hash
    with Session(engine) as session:
        merged_x = session.merge(x)
        session.commit()
        session.refresh(merged_x)


def test_graph_insert():
    g = Graph(graph="import hi", prompt="LO = 'avavav'")
    go(g)
    print(g.id)

def DANGER_DANGER_DANGER_read_graph_from_a_folder(folder: str):
    graph_file = os.path.join(folder, "graph.py")
    prompt_file = os.path.join(folder, "prompt.py")
    with open(graph_file, "r") as f:
        graph = f.read()
    with open(prompt_file, "r") as f:
        prompt = f.read()
    graph = Graph(graph=graph, prompt=prompt)

    # remove all graphs in db
    with Session(engine) as session:
        session.exec(delete(Graph))
        session.commit()

    go(graph)

def DANGER_DANGER_DANGER_test_read_graph_from_a_folder():
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))
    DANGER_DANGER_DANGER_read_graph_from_a_folder("sample/cot")
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))


def read_task_from_a_parquet(filepath: str):
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

def test_read_task_from_a_parquet():
    with Session(engine) as session:
        print(len(session.exec(select(Task)).all()))
    read_task_from_a_parquet("/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet")
    with Session(engine) as session:
        print(len(session.exec(select(Task)).all()))

def log_experiment(graph_id: bytes, task_id: bytes, localvar: dict, output: str, answer: str):
    from image_shelve import put_log
    localvar['__OUTPUT__'] = output
    localvar['__ANSWER__'] = answer
    log_id = put_log(localvar)
    with Session(engine) as session:
        run = Run(graph_id=graph_id, task_id=task_id, log_id=log_id, correct=(answer == output))
        session.add(run)
        session.commit()

if __name__ == "__main__":
    DANGER_DANGER_DANGER_test_read_graph_from_a_folder()
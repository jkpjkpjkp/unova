from sqlmodel import Field, SQLModel, create_engine, Session, select
import hashlib
import os
from image_shelve import go as img_go
db_name = "main.db"

class Graph(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph: str
    prompt: str

    @property
    def hash(self):
        if self.id:
            return self.id
        self.graph = self.graph.strip(' \n')
        self.prompt = self.prompt.strip(' \n')
        code = self.graph + '\n' + self.prompt
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Task(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    task: str
    answer: float

    @property
    def hash(self):
        if self.id:
            return self.id
        code = self.task + f"\\boxed{{{self.answer}}}"
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

class Run(SQLModel, table=True):
    id: bytes = Field(primary_key=True)
    graph_id: bytes = Field(foreign_key="graph.id")
    task_id: bytes = Field(foreign_key="task.id")
    log_id: bytes
    correct: bool

    @property
    def hash(self):
        if self.id:
            return self.id
        code = str(self.graph_id) + '\n' + str(self.task_id) + '\n' + str(self.log_id) + '\n' + str(self.correct)
        self.id = hashlib.sha256(code.encode('utf-8')).digest()
        return self.id

engine = None

def init():
    global engine
    engine = create_engine(f"sqlite:///{db_name}")
    SQLModel.metadata.create_all(engine)

def go(x):
    x.id = x.id or x.hash
    with Session(engine) as session:
        merged_x = session.merge(x)
        session.commit()
        session.refresh(merged_x)

def graph_stat(x: Graph):
    x.id = x.id or x.hash
    with Session(engine) as session:
        runs = session.exec(select(Run).where(Run.graph_id == x.id))
        tasks = {}
        for run in runs:
            if run.task_id not in tasks:
                tasks[run.task_id] = []
            tasks[run.task_id].append(run)
        correct = 0
        for task in tasks:
            correct += sum([run.correct for run in tasks[task]]) / len(tasks[task])
        return correct, len(tasks)

def task_stat(x: Task):
    x.id = x.id or x.hash
    with Session(engine) as session:
        runs = session.exec(select(Run).where(Run.task_id == x.id))
        return sum(map(lambda x: x.correct, runs)), len(runs)
    




def test_graph_insert():
    g = Graph(graph="import hi", prompt="LO = 'avavav'")
    go(g)
    print(g.id)

def read_graph_from_a_folder(folder: str):
    graph_file = os.path.join(folder, "graph.py")
    prompt_file = os.path.join(folder, "prompt.py")
    with open(graph_file, "r") as f:
        graph = f.read()
    with open(prompt_file, "r") as f:
        prompt = f.read()
    graph = Graph(graph=graph, prompt=prompt)
    go(graph)

def test_read_graph_from_a_folder():
    with Session(engine) as session:
        print(len(session.exec(select(Graph)).all()))
    read_graph_from_a_folder("sample/cot")
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

if __name__ == "__main__":
    init()
    test_read_task_from_a_parquet()
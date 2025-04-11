from sqlmodel import Field, SQLModel, create_engine, Session, select
import hashlib
db_name = "main.db"

class Graph(SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph: str
    prompt: str

    @property
    def hash(self):
        if self.id:
            return self.id
        self.graph = self.graph.strip(chars=' \n')
        self.prompt = self.prompt.strip(chars=' \n')
        code = self.graph + '\n' + self.prompt
        self.id = hashlib.sha256(code)
        return self.id


class Task(SQLModel, table=True):
    id: int = Field(primary_key=True)
    task: str
    answer: float | str


    @property
    def hash(self):
        if self.id:
            return self.id
        code = self.task + f"\\boxed{{{self.answer}}}"
        self.id = hashlib.sha256(code)
        return self.id

class Run(SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph_id: int = Field(foreign_key="graph.id")
    task_id: int = Field(foreign_key="task.id")
    log: dict
    correct: bool

engine = None

def init():
    global engine
    engine = create_engine(f"sqlite:///{db_name}")
    SQLModel.metadata.create_all(engine)

def go(x):
    x.id = x.id | x.hash
    with Session(engine) as session:
        session.add(x)
        session.commit()

def graph_stat(x: Graph):
    x.id = x.id | x.hash
    with Session(engine) as session:
        session.exec(select(Run).where(Run.graph_id == x.id))
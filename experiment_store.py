from sqlmodel import Field, SQLModel, create_engine, Session, select
import hashlib
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




if __name__ == "__main__":
    init() # Initialize the database engine
    test_graph_insert()
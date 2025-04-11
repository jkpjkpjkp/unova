from sqlmodel import Field, SQLModel, create_engine
db_name = "main.db"

class Graph(SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph: str
    prompt: str

class Task(SQLModel, table=True):
    id: int = Field(primary_key=True)
    task: str

class Run(SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph_id: int = Field(foreign_key="graph.id")
    task_id: int = Field(foreign_key="task.id")
    log: dict
    result: bool


def init():
    engine = create_engine(f"sqlite:///{db_name}")
    SQLModel.metadata.create_all(engine)
    

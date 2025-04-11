from typing import Any, Dict, List, Optional, Iterator
from sqlmodel import Field, SQLModel
import polars as pl
from PIL import Image
from io import BytesIO
from image_shelve import store
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
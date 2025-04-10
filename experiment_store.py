from typing import Any, Dict, List, Optional, Iterator
from sqlalchemy import create_engine, select, func, JSON, ForeignKey, String, Table, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import polars as pl
from PIL import Image
from io import BytesIO
from image_shelve import store
db_name = "experiment.db"

class Base(DeclarativeBase):
    pass

class Graph(Base):
    __tablename__ = "graphs"
    id: Mapped[int] = mapped_column(primary_key=True)
    graph: Mapped[str] = mapped_column(nullable=False)
    prompt: Mapped[str] = mapped_column(nullable=False)
    runs: Mapped[list["Run"]] = relationship(back_populates="graph")
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=True)

class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    task: Mapped[str] = mapped_column(nullable=False)
    runs: Mapped[list["Run"]] = relationship(back_populates="task")
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=True)

class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    graph_id: Mapped[int] = mapped_column(ForeignKey("graphs.id"))
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    locals: Mapped[dict] = mapped_column(JSON, nullable=True)
    judge: Mapped[dict] = mapped_column(JSON, nullable=True)
    correct: Mapped[bool] = mapped_column(nullable=False)
    graph: Mapped["Graph"] = relationship(back_populates="runs")
    task: Mapped["Task"] = relationship(back_populates="runs")

engine = create_engine(f"sqlite:///{db_name}")
Base.metadata.create_all(engine)

def import_parquet_tasks(path: str):
    df = pl.read_parquet(path)
    with Session(engine) as session:
        existing_tasks = {task.task for task in session.query(Task.task).all()}
        for row in df.iter_rows(named=True):
            question = row['question_text']  # Adjust column name if different
            images = [Image.open(BytesIO(img['bytes'])) for img in row['question_images_decoded']]
            images_str = ' '.join(store(images))
            task_str = images_str + question
            if task_str in existing_tasks:
                continue
            session.add(Task(task=task_str))
            existing_tasks.add(task_str)  # Update the set with the new task
        session.commit()

def test_import_parquet_tasks():
    import_parquet_tasks('/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet')
    with Session(engine) as session:
        tasks = session.query(Task).limit(5).all()
        print("First 5 tasks:")
        for task in tasks:
            print(task.task)
        
        # Check total number of tasks
        total_tasks = session.query(Task).count()
        print(f"Total tasks in database: {total_tasks}")


test_import_parquet_tasks()


def all_tasks() -> Iterator[Task]:
    with Session(engine) as session:
        return session.query(Task).all()

def log_experiment(graph_id: int, task_id: int, localvar: dict, output: str, answer: str):
    with Session(engine) as session:
        session.add(Run(graph_id=graph_id, task_id=task_id, locals=localvar, output=output, answer=answer, correct=(answer == answer)))
        session.commit()


def calc_win_rate(graph: Optional[Graph], task: Optional[Task]) -> float:
    with Session(engine) as session:
        if graph and task:
            query = select(Run).where(Run.graph == graph, Run.task == task)
            runs = session.execute(query).scalars().all()
            return sum(run.correct for run in runs) / len(runs)
        elif graph:
            query = select(Run).where(Run.graph == graph)
            runs = session.execute(query).scalars().all()
            return sum(run.correct for run in runs) / len(runs)
        elif task:
            query = select(Run).where(Run.task == task)
            runs = session.execute(query).scalars().all()
            return sum(run.correct for run in runs) / len(runs)
        else:
            raise ValueError("No graph or task provided")
            



###### UNDONE BELOW

def get_all_graph_stats() -> List[Dict[str, Any]]:
    """
    Compute columnar statistics for all graphs: number of distinct tasks and average success rate.
    
    Returns:
        List of dictionaries, each containing:
        - graph_id (int): The ID of the graph.
        - num_tasks (int): Number of distinct tasks run with the graph.
        - avg_success_rate (float or None): Average success rate across tasks, or None if no tasks.
    """
    with Session(engine) as session:
        # Subquery: Compute success rate (avg of correct) for each graph-task pair
        subquery = (
            select(
                Run.graph_id,
                Run.task_id,
                func.avg(Run.correct).label('success_rate')
            )
            .group_by(Run.graph_id, Run.task_id)
            .subquery()
        )
        
        # Main query: Count distinct tasks and average success rates per graph
        query = (
            select(
                Graph.id.label('graph_id'),
                func.count(subquery.c.task_id).label('num_tasks'),
                func.avg(subquery.c.success_rate).label('avg_success_rate')
            )
            .outerjoin(subquery, Graph.id == subquery.c.graph_id)
            .group_by(Graph.id)
        )
        
        # Execute and format results
        results = session.execute(query).all()
        return [
            {
                'graph_id': row.graph_id,
                'num_tasks': row.num_tasks,
                'avg_success_rate': row.avg_success_rate
            }
            for row in results
        ]
def test_get_all_graph_stats():
    stats = get_all_graph_stats()

    print("Graph ID | Num Tasks | Avg Success Rate")
    print("---------|-----------|-----------------")
    for stat in stats:
        avg_rate = f"{stat['avg_success_rate']:.4f}" if stat['avg_success_rate'] is not None else "N/A"
        print(f"{stat['graph_id']:8d} | {stat['num_tasks']:9d} | {avg_rate}")
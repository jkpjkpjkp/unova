from typing import *
from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

class Base(DeclarativeBase):
    pass

class Graph(Base):
    __tablename__ = "graphs"

    id: Mapped[int] = mapped_column(primary_key=True)
    graph: Mapped[str] = mapped_column(nullable=False)
    prompt: Mapped[str] = mapped_column(nullable=False)
    runs: Mapped[list["Run"]] = relationship(back_populates="graph")
    

class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    task: Mapped[str] = mapped_column(nullable=False)
    runs: Mapped[list["Run"]] = relationship(back_populates="task")

class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    graph: Mapped["Graph"] = relationship(back_populates="runs")
    task: Mapped["Task"] = relationship(back_populates="runs")
    locals: Mapped[dict] = mapped_column(nullable=True)
    judge: Mapped[dict] = mapped_column(nullable=True)
    correct: Mapped[bool] = mapped_column(nullable=False)


engine = create_engine("sqlite:///experiment.db")
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
            

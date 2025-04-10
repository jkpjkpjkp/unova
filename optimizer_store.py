from typing import Any, Dict, List, Optional, Iterator
from sqlalchemy import create_engine, select, func, JSON, ForeignKey, String, Table, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import polars as pl
from PIL import Image
from io import BytesIO
from image_shelve import store
db_name = "optimizer.db"

class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "optimization_run"
    id: Mapped[int] = mapped_column(primary_key=True)
    parents_ids: Mapped[List[int]] = mapped_column()
    task_ids: Mapped[List[int]] = mapped_column()
    log_ids: Mapped[List[int]] = mapped_column()
    response: Mapped[str] = mapped_column()
    child_id: Mapped[int] = mapped_column()
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=True)


def pick_pair():


def optimu():



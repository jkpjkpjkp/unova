import polars as pl
import os
from sqlalchemy.ext.asyncio import create_async_engine

filename = 'db/main.sqlite'
tablename = 'zero'
split = 'zerobench_subquestions'

db_path = os.path.abspath(filename)
connection_uri = f"sqlite:///{db_path}"
print(db_path)

def store(df=None):
    splits = {
        'zerobench': 'data/zerobench-00000-of-00001.parquet',
        'zerobench_subquestions': 'data/zerobench_subquestions-00000-of-00001.parquet'
    }
    df = df or pl.read_parquet('hf://datasets/jonathan-roberts1/zerobench/' + splits[split])
    df.write_database(table_name=tablename, connection=connection_uri, if_table_exists='replace')

def manipulate():
    df = load()
    df = df.drop('image_attribution')
    df = df.with_columns(
        ("z_" + pl.col('question_id')).alias('uid')
    )
    store(df)

def load():
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    return pl.read_database(
        query=f"SELECT * FROM {tablename}",
        connection=async_engine,
    )

# Execute the functions
store()  # Create the database first
df = load()  # Then load the data
print(df.columns)
print(df.head(10))
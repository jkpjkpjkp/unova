import polars as pl

filename = 'main.sqlite'
tablename = 'zero'
split = 'zerobench_subquestions'

def store():
    # Login using e.g. `huggingface-cli login` to access this dataset
    splits = {'zerobench': 'data/zerobench-00000-of-00001.parquet', 'zerobench_subquestions': 'data/zerobench_subquestions-00000-of-00001.parquet'}
    df = pl.read_parquet('hf://datasets/jonathan-roberts1/zerobench/' + splits[split])
    df.write_database(table_name=tablename, connection=f"sqlite:///{filename}")

def load():
    return pl.read_database_uri(f"SELECT * FROM {tablename}", uri=f"sqlite:///{filename}")

# store()
df = load()
print(df.columns)
print(df.head(10))
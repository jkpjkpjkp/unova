import polars as pl
import os
from sqlalchemy.ext.asyncio import create_async_engine
from PIL import Image
import io

filename = 'main.sqlite'
tablename = 'zero'
split = 'zerobench_subquestions'

db_path = os.path.abspath(filename)
connection_uri = f"sqlite:///{db_path}"
print(db_path)

def store():
    splits = {
        'zerobench': 'data/zerobench-00000-of-00001.parquet',
        'zerobench_subquestions': 'data/zerobench_subquestions-00000-of-00001.parquet'
    }
    df = pl.read_parquet('hf://datasets/jonathan-roberts1/zerobench/' + splits[split])
    df.write_database(table_name=tablename, connection=connection_uri, if_table_exists='replace')

def load():
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    return pl.read_database(
        query=f"SELECT * FROM {tablename}",
        connection=async_engine,
    )

# Execute the functions
# store()  # Create the database first
df = load()  # Then load the data
print(df.columns)
print(df.head(10))

# Sanity check: display the first image from 'question_images_decoded'
first_image_data = df['question_images_decoded'][0]
try:
    image = Image.open(io.BytesIO(first_image_data))
    image.show()
except Exception as e:
    print(f"Error displaying image: {e}")
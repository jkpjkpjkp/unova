import polars as pl
from PIL import Image
from io import BytesIO

df = pl.read_parquet('/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet')

print(df.head())

def display_image():
    first_row = df.row(0, named=True)
    images = first_row['question_images_decoded'] # list of images

    for image in images:
        print(image.keys())
        image = Image.open(BytesIO(image['bytes']))
        image.show()


holy_grail = pl.read_ndjson('/mnt/home/jkp/hack/tmp/MetaGPT/counting_zero.jsonl')

print(holy_grail.head())

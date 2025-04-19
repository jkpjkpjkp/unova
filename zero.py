import polars as pl

df = pl.read_parquet('/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet')

print(df.columns) # ['question_id', 'question_text', 'question_images_decoded', 'question_answer', 'question_images', 'image_attribution']
print(df.head())
print(df.row(0, named=True).keys())

def display_image():
    first_row = df.row(0, named=True)
    images = first_row['question_images_decoded'] # list of images
    print(type(images), len(images), type(images[0]))
    print(images[0].keys(), len(images[0]['bytes']))
    print(images[0]['bytes'][:30], images[0]['path'])

    # for image in images:
    #     print(image.keys())
    #     image = Image.open(BytesIO(image['bytes']))
    #     image.show()

display_image()

# holy_grail = pl.read_ndjson('/mnt/home/jkp/hack/tmp/MetaGPT/counting_zero.jsonl')

# print(holy_grail.head())

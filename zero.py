import polars as pl
from PIL import Image
import io

df = pl.read_parquet('/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet')


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
if __name__ == "__main__":
    print(df.columns) # ['question_id', 'question_text', 'question_images_decoded', 'question_answer', 'question_images', 'image_attribution']
    print(df.head())
    print(df.row(0, named=True).keys())
    display_image()

# holy_grail = pl.read_ndjson('/mnt/home/jkp/hack/tmp/MetaGPT/counting_zero.jsonl')

# print(holy_grail.head())

def get_task_data(task_id):
    filtered_df = df.filter(pl.col('question_id') == task_id)
    
    assert filtered_df.height == 1, f"Task ID {task_id} not found or duplicate."

    row = filtered_df.row(0, named=True)
    
    question_text = row['question_text']

    images = row['question_images_decoded']
    
    assert len(images) == 1, f"Task ID {task_id} does not have exactly one image."
    
    image_data = images[0]['bytes']
    
    image = Image.open(io.BytesIO(image_data))
    
    return (image, question_text)
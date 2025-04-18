BBOX = """given an image and a question, your task is to cut out relevant parts in the image.
specifically, you should output the coordinates of the bounding box of region relevant to the question.
make sure that the question is answerable with only the bbox retaining. 

question: {question}

crop the area related to the question in curly braces like this: {{x1, y1, x2, y2}}, e.g. {{0, 0, 1000, 1000}}. (x1, y1, x2, y2 are normalized the coordinates of the bounding box, in [0, 1000])
"""

MASK_CAPTION = """You are a helpful assistant that captions masks.
please caption the masked area. notice, only the 1st image is the object you should describe. the other images are just for context, and should not be described.

please think step by step, and place your final caption in curly braces like this: {your_caption}.
"""


COT = "\n\n\nLet’s think step by step and give the final answer in curly braces, like this: {{final_answer}}"

AUGMENTATION_EXPLENATION = """{question}
these are some sam masks and related caption in this image.
each mask contains a caption, pixel coordinate, and depth. 
{masks}


Let’s think step by step and give the final answer in curly braces, like this: {{final_answer}}
"""
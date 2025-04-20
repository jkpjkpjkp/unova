BBOX = """given an image and a question, your task is to cut out relevant parts in the image.
specifically, you should output the coordinates of the bounding box of region relevant to the question.
make sure that the question is answerable with only the bbox retaining. 

question: {question}

crop the area related to the question in curly braces like this: {{x1, y1, x2, y2}}, e.g. {{0, 0, 1000, 1000}}. (x1, y1, x2, y2 are normalized the coordinates of the bounding box, in [0, 1000])
"""

INFO_SPREAD="""I will explain to you your current task.  we are trying to answer a vqa question “{question}”.  but instead of presenting you with the whole image, we decided to partition the image and here is only part of it {image}.  we will later aggregate responses from all parts, in which what you are about to say is one part.

So please provide concise information for ultimately answering the question, in curly braces {{helpful information}}. if the image part assigned to you contains no relevant information, conclude with empty curly braces {{}}. 

Let's think step by step and give the final visual-information extraction in curly braces like this: {{information}}
"""


INFO_GATHER = """I will explain to you your current task.  we are trying to answer a vqa question “{question}”.  but instead of presenting you with the whole image, we have partitioned the image and present each part to one agent. Now is time to aggregate responses from all parts.

Here are the responses from all agents each presented with an un-overlapping part of the image.

{parts}


please answer the original vqa question.

let's think step by step and give the final answer in curly braces like this: {{final answer}}
{SoM_image}
"""

COT = "\n\n\nLet’s think step by step and give the final answer in curly braces, like this: {{final_answer}}"
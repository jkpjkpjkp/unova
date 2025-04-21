
COT = """original_image: {original_image}
set_of_mask: {set_of_mask_image}
{question}
\n\n\nLet’s think step by step and give the final answer in curly braces, like this: {{final_answer}}"""


prompt_dict = {
    'COT': COT,
}
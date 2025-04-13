MERGE_PROMPT = 'We are optimizing llm workflows. Below are 2 workflows and 2 tasks, each workflow is correct on exactly one task. I want you to analyze why for each task one got it right and the other wrong. After the analysis I desire you to come up with a third workflow combining the strengths of the two, that will be correct on both tests. This workflow should be written with Integrity, that is, it will not give away specifics of the two tests it will face.'
MERGE_PROMPT_TAIL = '<graph1>{graph1}</graph1><graph2>{graph2}</graph2><task1>{task1}</task1><task2>{task2}</task2><graph1_on_task_1(correct)>{graph1_on_task_1_correct}</graph1_on_task_1(correct)><graph2_on_task_1(wrong)>{graph2_on_task_1_wrong}</graph2_on_task_1(wrong)><graph1_on_task_2(wrong)>{graph1_on_task_2_wrong}</graph1_on_task_2(wrong)><graph2_on_task_2(correct)>{graph2_on_task_2_correct}</graph2_on_task_2(correct)>'

MERGE_PROMPT += '\n' + MERGE_PROMPT_TAIL






CUSTOM_USE = r"""
Here's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>XXX_PROMPT="your_prompt"</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=self.prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated multimodal results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**

Your response should be in the following format:
<modification>xxx(describe what you plan to modify)</modification>
<graph>xxx(graph code)</graph>
<prompt>xxx(prompt code)</prompt>
in accordance with reference sample. 
"""


MERGE_PROMPT += AFLOW_TAIL
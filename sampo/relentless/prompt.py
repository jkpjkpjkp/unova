CORRECT_IT = """this is a difficult vqa question alongside its answer.

Question: {question}
Answer: {answer}

Try to analyze based on the ground truth answer the reasoning that led to it. 
after thinking about why the answer might be this, please modify the graph to a workflow that can come up with this correct answer (of course, without giving it away in the prompts. ) 


Here is a graph example and the corresponding prompt (prompt only related to the custom method). You must write your graph that can solve the above task in the same format as this graph. \n
<sample>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>
</sample>
is a log on how this graph failed to answer the question correctly, which can be used as references for optimization:
{log}

First, provide optimization ideas. 
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operators and prompts, which have already been automatically imported.
"""


CUSTOM_USE = """
Here's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>XXX_PROMPT="your_prompt"</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=self.prompts.XXX_PROMPT)
# You can also concatenate previously generated multimodal results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n
"""
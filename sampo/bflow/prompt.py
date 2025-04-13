AFLOW = """You are building a Graph and corresponding Prompt to jointly solve {type} problems. 
Referring to the given graph and prompt, which forms a basic example of a {type} solution approach, 
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your 
single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When 
optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider 
Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), 
or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph 
complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical representation.
Output the modified graph and all the necessary Prompts in prompt.py (if needed).
The prompt you need to generate is only the one used in `XXX_PROMPT` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
the generated prompt must not contain any placeholders.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process.


Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <modification>(such as:add /delete /modify / ...)</modification>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and **no more than 5 lines of code may be changed per modification**—extensive modifications are strictly prohibited to maintain project focus!
Sometimes it is a very good idea to shrink code and remove unnecessary steps. 
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, and MM (short for MultimodalMessage), which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**

"""
from experiment_store import Graph, RRun, engine, Run, Task, Optimi
from image_shelve import callopenai
import re
from sqlmodel import Session, select
import math

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

def aflow(run: RRun) -> Graph:
    prompt = AFLOW.format(
        type = run.graph.task_tag,
        graph = run.graph.graph,
        prompt = run.graph.prompt,
        operator_description = r""""Custom": {
    "description": "A basic operator that executes custom instructions on input",
    "interface": "async def __call__(self, input, instruction)"
  },""",
        log = run.log,
    )
    response = callopenai(prompt)
    return Graph(
        graph = re.match(r"<graph>(.*?)</graph>", response, re.DOTALL).group(1),
        prompt = re.match(r"<prompt>(.*?)</prompt>", response, re.DOTALL).group(1),
        task_tag = run.graph.task_tag,
    )



##### UNDONE BELOW

def select_run() -> RRun:
    C = 0.1
    with Session(engine) as session:
        runs = session.exec(select(Run)).all()
        graphs = session.exec(select(Graph)).all()
        tasks = session.exec(select(Task)).all()
        optims = session.exec(select(Optimi)).all()
    N = len(optims)
    graph_stat = {g.id: {} for g in graphs}
    task_stat = {t.id: [] for t in tasks}
    for run in runs:
        if run.task_id not in graph_stat[run.graph_id]:
            graph_stat[run.graph_id][run.task_id] = []
        graph_stat[run.graph_id][run.task_id].append(run.correct)
        task_stat[run.task_id].append(run.correct)
    
    for graph_id in graph_stat:
        corr = 0
        tot = 0
        for task_id in graph_stat[graph_id]:
            corr += sum(graph_stat[graph_id][task_id]) / len(graph_stat[graph_id][task_id])
            tot += 1
        graphs.append((corr+1, tot+2, graph_id))
    graph_scores = [(x[0]/x[1]) + C * math.sqrt(2 * math.log(N) / x[1]) for x in graphs]
    argmax_graph = graphs[graph_scores.index(max(graph_scores))]
    if not graph_stat[argmax_graph.id][1]:
        from graph import let_us_pick
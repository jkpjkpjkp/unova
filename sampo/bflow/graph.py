from experiment_store import Graph, RRun, engine, Run, Task, Optimi
from image_shelve import callopenai
import re
from sqlmodel import Session, select
import math

def aflow(run: RRun, prompt_custom) -> Graph:
    prompt = prompt_custom['AFLOW'].format(
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
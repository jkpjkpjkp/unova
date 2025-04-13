from experiment_store import Graph, RRun, engine, Run, Task, Opti, Ron
from image_shelve import callopenai, get_log
import re
from sqlmodel import Session, select
import math

def aflow(run: Run, prompt_custom) -> Graph:
    with Session(engine) as session:
        rons, runs = session.select(Ron, Run).where(Ron.run_ids[0] == Run.id and Run.graph_id == run.graph_id).all()
    experience = ['Absolutely prohibit ' + get_log(x.log_id)['modification'] for x in rons]
    experience += '\n'.join(experience)
    experience = 'These are some things we tried before. for diversity please try something else: \n' + experience
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
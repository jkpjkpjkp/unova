import re
import functools
import sys
from image_shelve import callopenai, put_log
import os
from experiment_store import Graph, Task, engine, Run, go, Opti, Ron, read_graph_from_a_folder
from tqdm import tqdm
import asyncio
from sqlmodel import Session, select
from typing import Tuple
import random
import math
from typing import Optional

async def operator_custom(input, instruction):
    prompt = instruction + input
    response = await callopenai(prompt)
    return response

operators_dict = {
    "Custom": operator_custom,
}


def extract_local_variables(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        original = sys.gettrace()
        captured_locals = {}
        def trace(frame, event, _arg):
            if event == 'return' and frame.f_code is func.__code__:
                nonlocal captured_locals
                captured_locals = frame.f_locals
            return trace
        sys.settrace(trace)
        try:
            result = await func(*args, **kwargs)
        except:
            sys.settrace(original)
            raise
        sys.settrace(original)
        captured_locals = dict(captured_locals)
        captured_locals.pop('self')
        return result, captured_locals
    return wrapper

def graph_executable(graph_code: str, prompt_code: str):
    graph_code += '\n' + prompt_code
    namespace = {}
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    graph = Graph(operators=operators_dict, prompt_custom=namespace)
    return extract_local_variables(graph.run)

async def llm_as_judge(output, answer):
    prompt = f"""
    You are a judge.
    You are given an output and a ground truth answer.
    You need to determine if the output is correct. 
    End your response with 1 if correct, 0 if incorrect. make sure this number is the final character of your response.
    Output: {output}
    Answer: {answer}
    """
    response = await callopenai(prompt)
    return int(response[-1]), prompt, response

async def run_(graph: Graph, task: Task):
    graph_executable = graph_executable(graph.graph, graph.prompt)
    output, localvar = await graph_executable(task.task)
    print(output)
    localvar['__OUTPUT__'] = output
    correct, prompt, response = await llm_as_judge(output, task.answer)
    localvar['__LLM_AS_A_JUDGE_PROMPT__'] = prompt
    localvar['__LLM_AS_A_JUDGE_RESPONSE__'] = response
    go(Run(graph=graph, task=task, log_id=put_log(dict(localvar)), correct=correct))
    return correct

def get_graph_runs(tag: Optional[str]) -> dict[bytes, dict[bytes, list[Run]]]:
    graph_stat = {g.id: {} for g in graphs}
    with Session(engine) as session:
        runs = session.exec(select(Run)).all()
        graphs = session.exec(select(Graph)).all()
    for run in runs:
        if run.task_id not in graph_stat[run.graph_id]:
            graph_stat[run.graph_id][run.task_id] = []
        graph_stat[run.graph_id][run.task_id].append(run.correct)

def get_graph_stat() -> dict[bytes, tuple[float, int]]:
    graph_stat = get_graph_runs()
    graphs = {}
    for graph_id in graph_stat:
        corr = 0
        tot = 0
        for task_id in graph_stat[graph_id]:
            corr += sum(graph_stat[graph_id][task_id]) / len(graph_stat[graph_id][task_id])
            tot += 1
        graphs[graph_id] = (corr+1, tot+2)
    return graphs

def get_task_runs() -> dict[bytes, list[Run]]:
    task_stat = {t.id: [] for t in tasks}
    with Session(engine) as session:
        runs = session.exec(select(Run)).all()
        tasks = session.exec(select(Task)).all()
    for run in runs:
        task_stat[run.task_id].append(run.correct)
    return task_stat

def get_task_stat() -> dict[bytes, tuple[float, int]]:
    task_stat = get_task_runs()
    tasks = {}
    for task_id in task_stat:
        tasks[task_id] = (sum(task_stat[task_id]) + 1, len(task_stat[task_id]) + 2)
    return tasks

async def let_us_pick(graph: Optional[Graph] = None) -> Tuple[Graph, Task]:
    if graph:
        graph_id = graph.id
    else:
        graphs = get_graph_stat()
        graph_id = random.choices(graphs.keys(), weights=[(stat[0] / stat[1]) ** 2 for stat in graphs.values()], k=1)[0]
    
    tasks = get_task_stat()
    task_id = random.choices(tasks.keys(), weights=[(0.4 - x[0]/x[1]) ** 2 for x in tasks.values()])[0]
    
    with Session(engine) as session:
        graph = session.exec(select(Graph).where(Graph.id == graph_id)).one()
        task = session.exec(select(Task).where(Task.id == task_id)).one()
    await run_(graph, task)


def extract_xml(str) -> dict:
    str = re.sub(r'^\s*<.*?>\s*', '', str)
    str = re.sub(r'\s*</.*?>\s*$', '', str)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(str)
    return {child.tag: child.text for child in root}

def ron_(opti: Opti, runs: list[Run]):
    opti_executable = graph_executable(opti.graph, opti.prompt)
    output, localvar = asyncio.run(opti_executable(runs))
    o_dic = extract_xml(output)
    new_graph = go(Graph(graph=o_dic['graph'], prompt=o_dic['prompt']))
    go(Ron(opti_id=opti.id, run_ids=runs, log_id=put_log(dict(localvar)), new_graph_id=new_graph.id))

def who_to_optimize(tag: Optional[str]) -> list[Run]:
    graph_runs = get_graph_runs()
    graph_stat = get_graph_stat()
    with Session(engine) as session:
        
     

async def run_graph_42():
    graph_folder = "/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7"
    # Read the graph once outside the loop if it's static
    graph = read_graph_from_a_folder(graph_folder)
    tasks = [let_us_pick(graph=graph) for _ in range(42)]
    results = await asyncio.gather(*tasks)
    # Optional: process results if needed
    print(f"Completed {len(results)} tasks.")

if __name__ == "__main__":
    asyncio.run(run_graph_42())
"""

takes in files (code text) and returns executable
"""


import re
import functools
import sys
from image_shelve import call_openai
import os
from experiment_store import Graph, Task, init, Run, log
from tqdm import tqdm
import asyncio
from sqlmodel import Session, select
from typing import Tuple
import experiment_store
import random

async def operator_custom(input, instruction):
    prompt = instruction + input
    response = await call_openai(prompt)
    return response


operators_dict = {
    "Custom": operator_custom,
}

def extract_graph_by_text(graph_load: str) -> str:
    pattern = re.compile(r"class Graph:.*?(?=^\S|\Z)", re.DOTALL | re.MULTILINE)
    return re.match(pattern, graph_load).group(0)

def extract_local_variables(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        original = sys.gettrace()
        captured_locals = {}
        def a(frame, event, _arg):
            if event == 'return' and frame.f_code is func.__code__:
                nonlocal captured_locals
                captured_locals = frame.f_locals
            return a
        sys.settrace(a)
        try:
            result = await func(*args, **kwargs)
        finally:
            sys.settrace(original)
        return result, captured_locals
    return wrapper

def extract_graph_by_exec(graph_code: str, prompt_code: str):
    graph_code += '\n' + prompt_code
    namespace = {}
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    graph = Graph(operators=operators_dict, prompt_custom=namespace)
    return extract_local_variables(graph.run)

def run(graph: Graph, task: Task):
    graph_executable = extract_graph_by_exec(graph.graph, graph.prompt)
    output, localvar = asyncio.run(graph_executable(task.task))
    answer = re.findall(r'\boxed{(.*?)}', output)[-1]
    log(graph.id, task.id, localvar, output, answer)
    return answer == task.answer

def let_us_pick() -> Tuple[Graph, Task]:
    with Session(experiment_store.engine) as session:
        runs = session.exec(select(Run)).all()
        graphs = session.exec(select(Graph)).all()
        tasks = session.exec(select(Task)).all()
    graph_stat = {g.id: {} for g in graphs}
    task_stat = {t.id: [] for t in tasks}
    for run in runs:
        if run.task_id not in graph_stat[run.graph_id]:
            graph_stat[run.graph_id][run.task_id] = []
        graph_stat[run.graph_id][run.task_id].append(run.correct)
        task_stat[run.task_id].append(run.correct)
    graphs = []
    for graph_id in graph_stat:
        corr = 0
        tot = 0
        for task_id in graph_stat[graph_id]:
            corr += sum(graph_stat[graph_id][task_id]) / len(graph_stat[graph_id][task_id])
            tot += 1
        graphs.append((corr+1, tot+2, graph_id))
    graph_scores = [(x[0]/x[1]) ** 2 for x in graphs]
    graph_scores = [x / sum(graph_scores) for x in graph_scores]
    graph_id = random.choices(graphs, weights=graph_scores, k=1)[0][2]
    tasks = []
    for task_id in task_stat:
        tasks.append((sum(task_stat[task_id]) + 1, len(task_stat[task_id]) + 2, task_id))
    task_scores = [(0.4 - x[0]/x[1]) ** 2 for x in tasks]
    task_scores = [x / sum(task_scores) for x in task_scores]
    task_id = random.choices(tasks, weights=task_scores, k=1)[0][2]
    with Session(experiment_store.engine) as session:
        graph = session.exec(select(Graph).where(Graph.id == graph_id)).one()
        task = session.exec(select(Task).where(Task.id == task_id)).one()
    return graph, task

def test_pick_then_execute():
    graph, task = let_us_pick()
    run(graph, task)


if __name__ == "__main__":
    experiment_store.init()
    for _ in range(42):
        test_pick_then_execute()
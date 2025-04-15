import re
import functools
import sys
from image_shelve import callopenai, put_log, model
import os
from db import Graph, Task, engine, Run, go, Groph, Ron, get_graph_from_a_folder, get, gett, count_rows, tag, graph as graph_, task as task_
from tqdm import tqdm
import asyncio
from sqlmodel import Session, select
from typing import Tuple
import random
import math
from typing import Optional

async def operator_custom(input, instruction=""):
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

def get_graph_executable(graph_code: str, prompt_code: str):
    graph_code += '\n' + prompt_code
    namespace = {}
    namespace['__name__'] = '__exec__'
    namespace['__package__'] = None
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    graph = Graph(operators=operators_dict, prompt_custom=namespace)
    return extract_local_variables(graph.run)

async def llm_judge(output, answer):
    prompt = f"""
    You are a judge.
    You are given an output and a ground truth answer.
    You need to determine if the output is correct. 
    put your final judgement in curly braces, like this: {{correct}} or {{incorrect}}.
    Output: {output}
    Answer: {answer}
    """
    response = await callopenai(prompt)
    try:
        match = re.findall(r"{{(.*?)}}", response)
        extracted_content = match.groups()[-1]
        assert extracted_content in ['correct', 'incorrect']
    except:
        print(f"No verdict found in {response}")
        extracted_content = 'incorrect'
    return extracted_content == 'correct', {'llm_judge_response': response}

async def judge(output, answer):
    try:
        match = re.findall(r"{{(.*?)}}", output)
        output = match.groups()[-1]
        if re.sub(r'\s+', '', output) == re.sub(r'\s+', '', answer):
            return True, {}
        elif any(c.isdigit() for c in output):
            response = await callopenai(f"is output and the answer the same? output: {output}, answer: {answer}. response in only 1 word, 'yes' (they are the same) or 'no'. ")
            return response.strip().lower()[:3] == 'yes', {'short_judge_response': response}
        else:
            return False, {}
    except Exception as e:
        print(e)
        return await llm_judge(output, answer)


async def run_(graph: Graph, task: Task):
    graph_executable = get_graph_executable(graph.graph, graph.prompt)
    output, localvar = await graph_executable(task.task)
    print(output)
    localvar['__OUTPUT__'] = output
    correct, info = await judge(output, task.answer)
    for k, v in info.items():
        localvar[k] = v
    return go(Run(graph=graph, task=task, log_id=put_log(dict(localvar)), correct=correct, tags=[model]))

def get_graph_stat() -> dict[bytes, tuple[float, int]]:
    graph_stat = gett(Run, Graph, Task, tag=tag)
    graphs = {graph_id:(
            sum( sum(x.correct for x in graph_stat[graph_id][task_id])/len(graph_stat[graph_id][task_id]) for task_id in graph_stat[graph_id] ) + 1, 
            len(graph_stat[graph_id])+2,
            ) for graph_id in graph_stat}
    return graphs

def get_task_stat() -> dict[bytes, tuple[int, int]]:
    task_stat = get(Run, Task, tag=tag)
    tasks = {}
    for task_id in task_stat:
        tasks[task_id] = (sum(x.correct for x in task_stat[task_id]) + 1, len(task_stat[task_id]) + 2)
    return tasks

async def let_us_pick(graph: Optional[Graph] = None) -> Tuple[Graph, Task]:
    if graph:
        graph_id = graph.id
    else:
        graphs = get_graph_stat()
        graph_id = random.choices(list(graphs.keys()), weights=[(x[0] / x[1]) ** 2 for x in graphs.values()])[0]
    
    tasks = get_task_stat()
    task_id = random.choices(list(tasks.keys()), weights=[max(0, 3 - x[1])**2 + max(0, 0.3 - x[0]/x[1]) for x in tasks.values()])[0]

    graph = graph_(graph_id)
    task = task_(task_id)
    return graph, task


def extract_xml(str) -> dict:
    str = re.sub(r'^\s*<.*?>\s*', '', str)
    str = re.sub(r'\s*</.*?>\s*$', '', str)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(str)
    return {child.tag: child.text for child in root}

async def ron_(groph: Groph, runs: list[Run]):
    groph_executable = get_graph_executable(groph.graph, groph.prompt)
    graph, localvar = await groph_executable(runs)
    new_graph = go(graph)
    ron_instance = Ron(groph_id=groph.id, run_ids=[x.id for x in runs], log_id=put_log(dict(localvar)), new_graph_id=new_graph.id)
    print(ron_instance)
    return go(ron_instance)

def who_to_optimize() -> Run:
    graph_runs = get(Run, Graph)
    graph_stat = get_graph_stat()
    graph_rons = get(Ron, Graph)
    task_stat = get_task_stat()
    total_rons = count_rows(Ron)
    uct_scores = {}
    C = math.sqrt(2)

    for graph_id, stat in graph_stat.items():
        num_correct, num_runs = stat
        win_rate = num_correct / num_runs
        exploration_term = C * math.sqrt(math.log(total_rons + 1) / (len(graph_rons[graph_id]) + 1))
        uct_scores[graph_id] = win_rate + exploration_term

    best_graph_id = max(uct_scores, key=uct_scores.get)
    runs_for_best_graph = graph_runs.get(best_graph_id, [])
    
    failed_runs = []
    for run in runs_for_best_graph:
        if not run.correct:
            failed_runs.append(run)
    def get_task_success_rate(run):
        stat = task_stat.get(run.task_id)
        if stat and stat[1] > 0:
            return stat[0] / stat[1]
        return 0.0
    return max(failed_runs, key=get_task_success_rate)

def test_who_to_optize():
    he = who_to_optimize()
    a = get_graph_from_a_folder("sampo/bflow", groph=True)
    asyncio.run(ron_(a, [he]))

async def run_graph_42(times: int = 42, judgement='llm', tag='zerobench'):
    # graph_folder = "/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7"
    graph_folder = "sample/basic"
    graph = get_graph_from_a_folder(graph_folder)
    tasks = [await let_us_pick(graph=graph) for _ in range(times)]
    results = await asyncio.gather(*[run_(graph, task, judgement=judgement) for graph, task in tasks])
    print(f"Completed {len(results)} tasks.")

if __name__ == "__main__":
    asyncio.run(run_graph_42(times=2, judgement='llm', tag='zerobench'))
    # a = get_graph_from_a_folder("sampo/bflow", groph=True)
    # for i in range(10):
    #     print(f"ROUND {i}")
    #     asyncio.run(run_graph_42(times=21))
    #     ron_(a, [who_to_optimize()])
    # test_who_to_optize()
    
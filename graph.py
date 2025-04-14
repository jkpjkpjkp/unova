import re
import functools
import sys
from image_shelve import callopenai, put_log, model
import os
from experiment_store import Graph, Task, engine, Run, go, Groph, Ron, get_graph_from_a_folder, get, gett, count_rows, tag, graph as graph_, task as task_
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



async def operator_crop(input, instruction="please indicate cropped area by (x1, y1, x2, y2), each in [0, 1000]"):
    prompt = instruction + input
    response = await callopenai(prompt)
    match = re.findall(r"Cropped area: \((.*?), (.*?), (.*?), (.*?)\)", response)
    if match:
        x1, y1, x2, y2 = map(int, match[0])
        return 
    else:
        raise Exception("[grok] No cropped area found")

operators_dict = {
    "Custom": operator_custom,
    "Crop": operator_crop,
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

def rule_judge(output, answer):
    try:
        match = re.findall(r"{{(.*?)}}", output)
        output = match.groups()[-1]
        return output == answer
    except:
        return False

async def llm_as_judge(output, answer):
    prompt = f"""
    You are a judge.
    You are given an output and a ground truth answer.
    You need to determine if the output is correct. 
    put your final scoring in curly braces, like this: {{1}} if correct, {{0}} if incorrect.
    Output: {output}
    Answer: {answer}
    """
    response = await callopenai(prompt)
    try:
        match = re.findall(r"{{(.*?)}}", response)
        extracted_content = match.groups()[-1]
        assert extracted_content == bool(extracted_content)
    except:
        print("No scoring found in {response}")
        extracted_content = 0
    return extracted_content, prompt, response

async def run_(graph: Graph, task: Task, judgement='llm'):
    graph_executable = get_graph_executable(graph.graph, graph.prompt)
    output, localvar = await graph_executable(task.task)
    print(output)
    localvar['__OUTPUT__'] = output
    if judgement == 'llm':
        correct, prompt, response = await llm_as_judge(output, task.answer)
        localvar['__LLM_AS_A_JUDGE_PROMPT__'] = prompt
        localvar['__LLM_AS_A_JUDGE_RESPONSE__'] = response
    elif judgement == 'rule':
        correct = rule_judge(output, task.answer)
    localvar['__MODEL__'] = model
    return go(Run(graph=graph, task=task, log_id=put_log(dict(localvar)), correct=correct, tags=[model]))

def get_graph_runs() -> dict[bytes, dict[bytes, list[Run]]]:
    return gett(Run, Graph, Task, tag=tag)

def get_graph_rons() -> dict[bytes, list[Ron]]:
    return get(Ron, Graph, tag=tag)

def get_graph_stat() -> dict[bytes, tuple[float, int]]:
    graph_stat = get_graph_runs()
    graphs = {graph_id:(
            sum( sum(x.correct for x in graph_stat[graph_id][task_id])/len(graph_stat[graph_id][task_id]) for task_id in graph_stat[graph_id] ) + 1, 
            len(graph_stat[graph_id])+2,
            ) for graph_id in graph_stat}
    return graphs

def get_task_runs() -> dict[bytes, list[Run]]:
    return get(Run, Task, tag=tag)

def get_task_stat() -> dict[bytes, tuple[int, int]]:
    task_stat = get_task_runs()
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

def ron_(groph: Groph, runs: list[Run]):
    groph_executable = get_graph_executable(groph.graph, groph.prompt)
    graph, localvar = asyncio.run(groph_executable(runs))
    new_graph = go(graph)
    go(Ron(groph_id=groph.id, opti_id=runs[0].id, run_ids=runs, log_id=put_log(dict(localvar)), new_graph_id=new_graph.id))

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
    ron_(a, [he])

async def run_graph_42(times: int = 42, judgement='llm', tag='zerobench'):
    # graph_folder = "/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7"
    graph_folder = "sample/basic"
    graph = get_graph_from_a_folder(graph_folder)
    tasks = [await let_us_pick(graph=graph) for _ in range(times)]
    results = await asyncio.gather(*[run_(graph, task, judgement=judgement) for graph, task in tasks])
    print(f"Completed {len(results)} tasks.")

if __name__ == "__main__":
    asyncio.run(run_graph_42(times=42, judgement='rule', tag='mmiq'))
    # a = get_graph_from_a_folder("sampo/bflow", groph=True)
    # for i in range(10):
    #     print(f"ROUND {i}")
    #     asyncio.run(run_graph_42(times=21))
    #     ron_(a, [who_to_optimize()])
    # test_who_to_optize()
    
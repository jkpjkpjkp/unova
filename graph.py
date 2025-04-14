import re
import functools
import sys
from image_shelve import callopenai, put_log
import os
from experiment_store import Graph, Task, engine, Run, go, Groph, Ron, read_graph_from_a_folder, get, gett, count_rows
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
    graph_executable = get_graph_executable(graph.graph, graph.prompt)
    output, localvar = await graph_executable(task.task)
    print(output)
    localvar['__OUTPUT__'] = output
    correct, prompt, response = await llm_as_judge(output, task.answer)
    localvar['__LLM_AS_A_JUDGE_PROMPT__'] = prompt
    localvar['__LLM_AS_A_JUDGE_RESPONSE__'] = response
    go(Run(graph=graph, task=task, log_id=put_log(dict(localvar)), correct=correct))
    return correct

def get_graph_runs() -> dict[bytes, dict[bytes, list[Run]]]:
    return gett(Run, Graph, Task)

def get_graph_rons() -> dict[bytes, list[Ron]]:
    return get(Ron, Graph)

def get_graph_stat() -> dict[bytes, tuple[float, int]]:
    graph_stat = get_graph_runs()
    graphs = {graph_id:(
            sum( sum(x.correct for x in graph_stat[graph_id][task_id])/len(graph_stat[graph_id][task_id]) for task_id in graph_stat[graph_id] ) + 1, 
            len(graph_stat[graph_id])+2,
            ) for graph_id in graph_stat}
    return graphs

def get_task_runs() -> dict[bytes, list[Run]]:
    return get(Run, Task)

def get_task_stat() -> dict[bytes, tuple[float, int]]:
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
    task_id = random.choices(list(tasks.keys()), weights=[0.25 -(0.5 - x[0]/x[1]) ** 2 for x in tasks.values()])[0]

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

def ron_(groph: Groph, runs: list[Run]):
    groph_executable = graph_executable(groph.graph, groph.prompt)
    print(groph_executable)
    output, localvar = asyncio.run(groph_executable(runs))

    o_dic = extract_xml(output)
    new_graph = go(Graph(graph=o_dic['graph'], prompt=o_dic['prompt']))
    go(Ron(groph_id=groph.id, run_ids=runs, log_id=put_log(dict(localvar)), new_graph_id=new_graph.id))

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
    a = read_graph_from_a_folder("sampo/bflow", groph=True)
    print(type(a))
    ron_(a, [he])

async def run_graph_42():
    # graph_folder = "/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7"
    graph_folder = "sample/basic"
    graph = read_graph_from_a_folder(graph_folder)
    tasks = [let_us_pick(graph=graph) for _ in range(42)]
    results = await asyncio.gather(*tasks)
    print(f"Completed {len(results)} tasks.")

if __name__ == "__main__":
    # a = read_graph_from_a_folder("sampo/bflow", groph=True)
    # for i in range(10):
    #     print(f"ROUND {i}")
    #     for _ in range(10):
    #         let_us_pick()
    #     he = who_to_optimize()
    #     ron_(a, [he])
    # test_who_to_optize()
    asyncio.run(run_graph_42())
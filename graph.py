"""

takes in files (code text) and returns executable
"""


import re
import functools
import sys
from image_shelve import call_openai
import os
from experiment_store import Graph, Task, all_tasks, log_experiment, calc_win_rate
from tqdm import tqdm
import asyncio

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
        original_trace = sys.gettrace()

        captured_locals = {}
        def trace_func(frame, event, _arg):
            if event == 'return' and frame.f_code is func.__code__:
                nonlocal captured_locals
                captured_locals = frame.f_locals
            return trace_func
        sys.settrace(trace_func)

        try:
            result = await func(*args, **kwargs)
        finally:
            sys.settrace(original_trace)
        return result, captured_locals
    return wrapper

def extract_graph_by_exec(graph_code: str, prompt_code: str):
    graph_code += '\n' + prompt_code
    namespace = {}
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    graph = Graph(operators=operators_dict, prompt_custom=namespace)
    return extract_local_variables(graph.run)


def conduct_experiment(graph: Graph, task: Task):
    graph_executable = extract_graph_by_exec(graph.graph, graph.prompt)
    output, localvar = asyncio.run(graph_executable(task.task))
    answer = re.findall(r'<boxed>(.*?)</boxed>', output)[-1]
    log_experiment(graph.id, task.id, localvar, output, answer)
    return answer == task.answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.graph_dir, 'graph.py'), 'r') as f:
        graph_code = f.read()
    with open(os.path.join(args.graph_dir, 'prompt.py'), 'r') as f:
        prompt_code = f.read()

    graph = Graph(prompt=prompt_code, graph=graph_code)
    tasks = all_tasks()
    print(len(tasks), 'tasks')
    for task in tqdm(tasks):
        conduct_experiment(graph, task)

    print(calc_win_rate(graph=graph))
import re
import functools
import sys
from image_shelve import callopenai
from db import Graph, Task, Run, go, Groph, Ron, get_graph_from_a_folder, get, get_strongest_graph, where
import asyncio
from typing import Tuple
import random
from typing import Optional
import argparse

async def operator_custom(input, instruction="", model='gemini-2.0-flash'):
    prompt = instruction + input
    response = await callopenai(prompt, model=model)
    return response

from image_shelve import extract_image as operator_extract_image
from image_shelve import img_go as operator_img_to_url
from image_shelve import sam2 as operator_sam2
from image_shelve import depth_estimator as operator_depth_estimator

operators_dict = {
    "Custom": operator_custom,
    "ExtractImage": operator_extract_image,
    "ImageToUrl": operator_img_to_url,
    "SAM2": operator_sam2,
    "Depth-Anything-V2": operator_depth_estimator,
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

def test_extract_local_variables():
    @extract_local_variables
    def test_func(a, b):
        return a + b
    result, localvar = test_func(1, 2)
    assert result == 3
    assert dict(localvar) == {'a': 1, 'b': 2}

def get_graph_executable(graph_code: str, prompt_code: str):
    graph_code += '\n' + prompt_code
    namespace = {}
    namespace['__name__'] = '__exec__'
    namespace['__package__'] = None
    
    if graph_code.endswith('"""'):
        pass
    elif graph_code.endswith('""'):
        graph_code += '"'
    elif graph_code.endswith('"'):
        graph_code += '""'
    elif graph_code.endswith('"":'):
        graph_code = graph_code[:-2] + '"""'
    try:
        exec(graph_code, namespace)
    except Exception:
        print("--- Executing graph code ---")
        print(graph_code)
        print("--- End graph code ---")
        raise
    Graph = namespace.get("Graph")
    graph = Graph(operators=operators_dict, prompts=namespace)
    return extract_local_variables(graph.run)

def extract_brace(x: str):
    match = re.search(r"{(.*?)}", x)
    return match.group(1)

def test_re():
    response = r'No verdict found in {incorrect}'
    assert extract_brace(response) == 'incorrect'

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
        extracted_content = extract_brace(response)
        assert extracted_content in ['correct', 'incorrect']
    except:
        print(f"No verdict found in {response}")
        extracted_content = 'incorrect'
    return extracted_content == 'correct', {'llm_judge_response': response}


async def judge(output, answer):
    try:
        matches = re.findall(r"{(.*?)}", output)
        if matches:
            extracted_output = matches[-1].strip()
            if re.sub(r'\s+', '', extracted_output) == re.sub(r'\s+', '', answer):
                return True, {}
            elif any(c.isdigit() for c in extracted_output):
                response = await callopenai(f"is output and the answer the same? output: {extracted_output}, answer: {answer}. response in only 1 word, 'yes' (they are the same) or 'no'. ")
                return response.strip().lower()[:3] == 'yes', {'short_judge_response': response}
            else:
                return False, {}
        else:
            print("No {...} found in output, falling back to LLM judge.")
            return await llm_judge(output, answer)

    except Exception as e:
        print(f"Error during regex/simple judging: {e}")
        return await llm_judge(output, answer)

def test_judge():
    assert asyncio.run(judge(r"{1}", "1")) == (True, {})

async def run_(graph: Graph, task: Task):
    graph_executable = get_graph_executable(graph.graph, graph.prompt)
    try:
        output, localvar = await graph_executable(task.task)
    except Exception as e:
        print(f"Error running graph {graph.id} on task {task.id}: {e}")
        raise
    print(output)
    localvar['__OUTPUT__'] = output
    correct, info = await judge(output, task.answer)
    for k, v in info.items():
        localvar[k] = v
    log_dict = {}
    for k, v in localvar.items():
        try:
            log_dict[k] = str(v)
        except Exception as e:
            print(f"Error converting log variable {k} to string: {e}")
            pass
    return go(Run(graph=graph, task=task, log=log_dict, correct=correct))

if __name__ == "__main__":
    asyncio.run(run_(get_graph_from_a_folder("sample/crop_seg"), Task(task="<image_x> What is the total number of pink rings forming the outer border of the heart shape?", answer="7")))
    exit()

def get_graph_stat(tag=None) -> dict[bytes, tuple[float, int]]:
    graph_stat = get(Run, Graph, Task, tag=tag)
    graphs = {graph:(
            sum( sum(x.correct for x in graph_stat[graph][task_id])/len(graph_stat[graph][task_id]) for task_id in graph_stat[graph] ) + 1, 
            len(graph_stat[graph])+2,
            ) for graph in graph_stat}
    return graphs

def test_get_graph_stat():
    print(len(get_graph_stat()))

def get_task_stat(tag=None) -> dict[bytes, tuple[int, int]]:
    task_stat = get(Run, Task, tag=tag)
    tasks = {}
    for task in task_stat:
        if not tag or tag in task.tags:
            tasks[task] = (sum(x.correct for x in task_stat[task]) + 1, len(task_stat[task]) + 2)
    return tasks

async def let_us_pick(graph: Optional[Graph] = None, num=1, tag=None) -> Tuple[Graph, Task]:
    if not graph:
        graphs = get_graph_stat()
        graph = random.choices(list(graphs.keys()), weights=[(x[0] / x[1]) ** 2 for x in graphs.values()])[0]
    
    tasks = get_task_stat(tag=tag)
    tasks = random.choices(list(tasks.keys()), weights=[max(0, 3 - x[1])**2 + max(0, 0.3 - x[0]/x[1]) for x in tasks.values()], k=num)
    return graph, tasks


def extract_xml(str) -> dict:
    str = re.sub(r'^\s*<.*?>\s*', '', str)
    str = re.sub(r'\s*</.*?>\s*$', '', str)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(str)
    return {child.tag: child.text for child in root}

async def ron_(groph: Groph, runs: list[Run], tag=None):
    groph_executable = get_graph_executable(groph.graph, groph.prompt)
    graph, localvar = await groph_executable(runs)
    if not graph.tags:
        graph.tags = []
    graph.tags.append(tag)
    new_graph = go(graph)
    log_str = {}
    for k, v in localvar.items():
        try:
            log_str[k] = str(v)
        except:
            pass
    return go(Ron(groph_id=groph.id, runs=runs, log=log_str, final_output=new_graph.id, tags=[tag]))


async def who_to_optimize(tag=None) -> Run:
    best_graph = get_strongest_graph()
    task_stat = get_task_stat(tag=tag)
    
    runs_for_best_graph = where(best_graph, Run, tag=tag)

    failed_runs = []
    for run in runs_for_best_graph:
        if not run.correct:
            failed_runs.append(run)
    if not failed_runs:
        print(f"No failed runs found for graph {best_graph.id} with tag '{tag}' among {len(runs_for_best_graph)} runs examined.")
        raise ValueError(f"No failed runs found in {len(runs_for_best_graph)} runs")
    
    def get_task_success_rate(run):
        stat = task_stat.get(run.task_id)
        if stat and stat[1] > 0:
            return stat[0] / stat[1]
        return 0.0
    
    return min(failed_runs, key=get_task_success_rate)

def test_who_to_optize():
    he = asyncio.run(who_to_optimize())
    a = get_graph_from_a_folder("sampo/bflow", groph=True)
    asyncio.run(ron_(a, [he]))

async def run_graph_42(graph: Graph, times: int = 42, tag=None):
    graph, tasks = await let_us_pick(graph=graph, num=times, tag=tag)
    results = await asyncio.gather(*[run_(graph, task) for task in tasks])
    print(f"Completed {len(results)} tasks.")

async def _aflow_iteration(groph, tag):
    graph, tasks_to_run = await let_us_pick(tag=tag, num=10)
    run_coroutines = [run_(graph, task) for task in tasks_to_run]
    results = await asyncio.gather(*run_coroutines)
    run_to_optimize = await who_to_optimize(tag=tag) 
    await ron_(groph, [run_to_optimize])
    return results

async def aflow(tag=None):
    a = get_graph_from_a_folder('sampo/bflow', groph=True)
    iteration_coroutines = [_aflow_iteration(a, tag) for _ in range(10)]
    await asyncio.gather(*iteration_coroutines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(aflow(args.tag))
    
"""

takes in files (code text) and returns executable
"""


import re
from typing import List, Literal
import functools
import sys
from pydantic import BaseModel, Field
from image_shelve import call_openai


###### UNDONE BELOW

async def operator_custom(input, instruction, mode: Literal["single_fill", "xml_fill", "code_fill"] = "single_fill"):
    prompt = instruction + input
    response = await call_openai(prompt)
    return response

###### UNDONE ABOVE


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
    namespace = {}
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    graph = Graph(operators=operators_dict, prompt_custom=namespace)
    return extract_local_variables(graph.run)
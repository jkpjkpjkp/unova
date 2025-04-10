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


async def custom(self, input, instruction, mode: Literal["single_fill", "xml_fill", "code_fill"] = "single_fill"):
    prompt = instruction + input
    response = await call_openai(prompt)
    return response

def extract_solve_graph(graph_load: str) -> List[str]:
    pattern = re.compile(r"class Graph:.*?(?=^\S|\Z)", re.DOTALL | re.MULTILINE)
    return re.findall(pattern, graph_load)

def load_graph(graph_code: str, prompt_code: str):
    namespace = {}
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    if Graph is None:
        raise ValueError("Graph not found in the provided code")
    graph = Graph(operators=operators_dict)
    return graph.run


def extract_local_variables(func):
    """
    Returns:
        a callable that
        Returns: (
            func's return, 
            captured_locals,
        }
    """
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





def main(code: str):
    
import re
from typing import List
import functools
import sys

def extract_solve_graph(graph_load: str) -> List[str]:
    pattern = re.compile(r"class Graph:.*?(?=^\S|\Z)", re.DOTALL | re.MULTILINE)
    return re.findall(pattern, graph_load)

def load_graph(graph_code: str, prompt_custom: str, operators_dict: dict):
    namespace = {}
    exec(graph_code, namespace)
    Graph = namespace.get("Graph")
    if Graph is None:
        raise ValueError("Graph not found in the provided code")
    graph = Graph(operators=operators_dict, prompt_custom=prompt_custom)
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

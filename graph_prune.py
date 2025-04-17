from db import get, Graph, callopenai

def graph_prune():
    graphs = get(Graph, tag_exclude='prompt no unnecessary detail')
    for graph in graphs:
        prompt = callopenai(f"Please remove unnecessarily detailed / specific examples from the following prompt: {graph.prompt}")
        graph.prompt = prompt

from action_node import operators
from sample.som.graph import Graph
from sample.som.prompt import prompt_dict
from zero import get_task_data
import asyncio
from db import MyStr

if __name__ == "__main__":
    task = get_task_data('37_3')
    graph = Graph(operators=operators, prompts={
        k: (v if not isinstance(v, str) else MyStr(v))
        for k, v in prompt_dict.items()
    })
    print(asyncio.run(graph.run((task['image'], task['question']))))

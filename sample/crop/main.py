from action_node import operators
from sample.crop.graph import Graph
from sample.crop.prompt import prompt_dict
from zero import get_task_data
from ve import VE
import asyncio

if __name__ == "__main__":
    image, task = get_task_data('42_2')
    ve = VE(image)
    graph = Graph(operators=operators, prompts=prompt_dict)
    print(asyncio.run(graph.run((ve, task))))

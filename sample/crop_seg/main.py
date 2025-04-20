from action_node import operators
from sample.crop_seg.graph import Graph
from zero import get_task_data
from sample.crop_seg.prompt import prompt_dict
import asyncio

if __name__ == "__main__":
    image, task = get_task_data('37_3')
    graph = Graph(operators=operators, prompts=prompt_dict)
    print(asyncio.run(graph.run((image, task))))

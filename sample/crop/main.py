from action_node import operators
from sample.crop.graph import Graph
from sample.crop.prompt import prompt_dict
from zero import get_task_data
import asyncio

if __name__ == "__main__":
    task = get_task_data('42_2')
    graph = Graph(operators=operators, prompts=prompt_dict)
    print(asyncio.run(graph.run((task['image'], task['question']))))

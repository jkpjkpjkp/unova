from action_node import operators
from sample.som.graph import Graph
from zero import get_task_data
from sample.som.prompt import prompt_dict
import asyncio

if __name__ == "__main__":
    task = get_task_data('37_3')
    graph = Graph(operators=operators, prompts=prompt_dict)
    print(asyncio.run(graph.run((task['image'], task['question']))))

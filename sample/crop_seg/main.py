from action_node import operators
from sample.crop_seg.graph import Graph
from zero import get_task_data
import asyncio

if __name__ == "__main__":
    image, task = get_task_data('42_2')
    graph = Graph(operators)
    print(asyncio.run(graph.run((image, task))))

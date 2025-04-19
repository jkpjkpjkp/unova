from db import *
import asyncio

tasks = get_high_variation_task(5)
graphs = get_strongest_graph(5)

queue = []
for graph in graphs:
    for task in task:
        queue.append(graph.run(task))

async def wait():
    await asyncio.gather(*queue)
asyncio.run(wait())


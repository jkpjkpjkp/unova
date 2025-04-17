from db import *
from image_shelve import *
from graph import *
import asyncio
import time


async def lavaroty():
    while True:
        graph = find_the_strongest_graph()
        tasks = find_hardest_tasks(10, tag='zerobench')
        await asyncio.gather(*[run_(graph, task) for task in tasks])
        time.sleep(10)
    

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # loop.run_until_complete(lavaroty())

async def ronron():
    mygroph = get_graph_from_a_folder('sampo/bflow', groph=True)
    run_to_optimize = await who_to_optimize(tag='zerobench') 
    await ron_(mygroph, [run_to_optimize])

async def fiveron():
    await asyncio.gather(*[ronron() for _ in range(5)])

if __name__ == "__main__":
    loop.run_until_complete(fiveron())
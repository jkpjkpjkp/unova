from db import *
from image_shelve import *
from graph import *
import asyncio
import time


async def lavaroty():
    for _ in range(1):
        graph = get_strongest_graph()
        tasks = find_hardest_tasks(2, tag='zerobench')
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

# if __name__ == "__main__":
#     loop.run_until_complete(fiveron())

async def run_graph_by_folder(folder: str):
    graph = get_graph_from_a_folder(folder)
    tasks = find_hardest_tasks(1, tag='zerobench')
    await asyncio.gather(*[run_(graph, task) for task in tasks])

# if __name__ == "__main__":
#     loop.run_until_complete(run_graph_by_folder('sample/crop_seg'))

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_graph_42(get_by_id(Graph, b"\xc1\xcf\x9b\x04\xe6\xa3\xef\xc0?6\x83\xd1\xe3\x0e''\xf85\xc5\x01\xe1exv\xd9\xb1\x8d\xe1\xb51o\xce"), tag='zerobench'))

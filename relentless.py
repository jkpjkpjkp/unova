import asyncio
from graph import run_, ron_, get_task_stat
from experiment_store import engine, Graph, graph, Task, Run, task, get_graph_from_a_folder, get, go
from sqlmodel import Session
import itertools
import random
from tqdm import tqdm

async def bombarda(run):
    with Session(engine) as session:
        session.add(run)
        graph_orig = run.graph
        task = run.task
        runs = [run]
        graph_ids = []
        relentless = get_graph_from_a_folder('sampo/bflow', groph=True)
        for _ in range(100):
            print('LEN: ', len(runs))
            ron = await ron_(relentless, [runs[0]])
            graph_ids.append(ron.new_graph_id)
            graph_ = graph(ron.new_graph_id)
            run = await run_(graph_, task)
            runs.append(run)
            if run.correct:
                break

def test_bob():
    ts = get_task_stat()
    ts = {k for k, v in ts.items() if v[0] == 1 and v[1] > 2 }

    cot_graph = get_graph_from_a_folder('sample/basic')
    tss = get(Run, Graph)[cot_graph.id]

    for t in tss:
        if t.id in ts:
            print('FOUND!')
            asyncio.run(bombarda(cot_graph, t.task))
            exit
    
    asyncio.run(bombarda(asyncio.run(run_(cot_graph, task(list(ts)[0])))))
    

    print(len(ts))

async def merge(graph1, graph2):
    runs = get(Run, Graph)
    run1 = runs[graph1.id]
    run2 = runs[graph2.id]
    tasks = list(set([x.task_id for x in itertools.chain(run1, run2) if not x.correct]))
    merg = get_graph_from_a_folder('sampo/merger', groph=True)
    go(merg)
    for _ in tqdm(range(100)):
        ron = await ron_(merg, [run1[0], run2[0]])
        new_graph = ron.new_graph
        go(new_graph)
        tasks = random.sample(tasks, 3)
        sum_correct = 0
        for task_id in tasks:
            current_task = task(task_id)
            run = await run_(new_graph, current_task)
            go(run)
            sum_correct += run.correct
        if sum_correct == 3:
            break

if __name__ == '__main__':
    test_bob()
    # merge(get_graph_from_a_folder('sample/basic'), get_graph_from_a_folder('/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7'))
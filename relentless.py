import asyncio
from graph import run_, ron_, get_task_stat
from experiment_store import engine, Graph, graph, Task, Run, task, get_graph_from_a_folder, get, go
from sqlmodel import Session
import itertools
import random
from tqdm import tqdm
def bombarda(run):
    with Session(engine) as session:
        session.add(run)
        graph_orig = run.graph
        task = run.task
        runs = [run]
        graph_ids = []
        relentless = get_graph_from_a_folder('sampo/bflow', groph=True)
        for _ in range(100):
            print('LEN: ', len(runs))
            ron = ron_(relentless, [runs[0]])
            graph_ids.append(ron.new_graph_id)
            graph_ = graph(ron.new_graph_id)
            run = run_(graph_, task)
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
            bombarda(cot_graph, t.task)
            exit
    
    bombarda(asyncio.run(run_(cot_graph, task(list(ts)[0]))))
    

    print(len(ts))

def merge(graph1, graph2):
    runs = get(Run, Graph)
    run1 = runs[graph1.id]
    run2 = runs[graph2.id]
    tasks = set([x.task_id for x in itertools.chain(run1, run2) if not x.correct])
    merg = get_graph_from_a_folder('sampo/merger', groph=True)
    go(merg)
    for _ in tqdm(range(100)):
        ron = ron_(merg, [run1[0], run2[0]])
        new_graph = ron.new_graph
        go(new_graph)
        tasks = random.sample(tasks, 3)
        sum = 0
        for task in tasks:
            run = run_(new_graph, task)
            go(run)
            sum += run.correct
        if sum == 3:
            break

if __name__ == '__main__':
    test_bob()
    # merge(get_graph_from_a_folder('sample/basic'), get_graph_from_a_folder('/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7'))
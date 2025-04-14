import asyncio
from graph import run_, ron_, get_task_stat
from experiment_store import engine, Graph, graph, Task, Run, task, read_groph_from_a_folder, read_graph_from_a_folder, get
from sqlmodel import Session

def bombarda(run):
    with Session(engine) as session:
        graph_orig = run.graph
        task = run.task
        runs = []
        graph_ids = []
        relentless = read_groph_from_a_folder('sampo/relentless')
        for _ in range(100):
            print('LEN: ', len(runs))
            ron = ron_(relentless, runs[0])
            graph_ids.append(ron.new_graph_id)
            graph_ = graph(ron.new_graph_id)
            run = run_(graph_, task)
            runs.append(run)
            if run.correct:
                break

if __name__ == '__main__':
    ts = get_task_stat()
    ts = {k for k, v in ts.items() if v[0] == 1 and v[1] > 2 }

    cot_graph = read_graph_from_a_folder('sample/basic')
    tss = get(Run, Graph)[cot_graph.id]

    for t in tss:
        if t.id in ts:
            print('FOUND!')
            bombarda(cot_graph, t.task)
            exit
    
    bombarda(asyncio.run(run_(cot_graph, task(list(ts)[0]))))
    

    print(len(ts))
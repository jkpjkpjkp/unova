from db import *
from db import _engine
import asyncio
from zero import get_task_data
import numpy as np
import json
import random
from pydantic import BaseModel
from typing import Callable

from aflow_prompt import *
from action_node import ActionNode, LLM, operators_doc

max_rounds = 25

def compute_probabilities(scores, alpha=0.2, lambda_=0.3):
    scores = np.array(scores, dtype=np.float64)
    n = len(scores)

    uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

    max_score = np.max(scores)
    shifted_scores = scores - max_score
    exp_weights = np.exp(alpha * shifted_scores)

    sum_exp_weights = np.sum(exp_weights)
    if sum_exp_weights == 0:
        raise ValueError("Sum of exponential weights is 0, cannot normalize.")

    score_prob = exp_weights / sum_exp_weights

    mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

    total_prob = np.sum(mixed_prob)
    if not np.isclose(total_prob, 1.0):
        mixed_prob = mixed_prob / total_prob

    return mixed_prob

def test_compute_probabilities():
    scores = [1.0, 2.0, 3.0]
    probs = compute_probabilities(scores)
    assert len(probs) == len(scores)
    assert np.isclose(np.sum(probs), 1.0)
    assert all(p >= 0 for p in probs)
    
    scores = [0.0, 0.0, 0.0]
    probs = compute_probabilities(scores)
    assert np.allclose(probs, [1/3, 1/3, 1/3])
    
    scores = [-1.0, -2.0, -3.0]
    probs = compute_probabilities(scores)
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    
    scores = [1.0, 2.0]
    probs = compute_probabilities(scores, alpha=0.5, lambda_=0.1)
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0)


async def experiment(
    num_graph=5,
    num_task=5,
):
    # selecting highest variation is self-stabilizing (good). 
    tasks = get_high_variation_task(num_task)
    graphs = get_strongest_graph(num_graph)
    print('Best Score: ', graphs[3].score)

    num_graph = len(graphs)
    assert num_graph > 0
    assert num_task == len(tasks)

    run = []
    valid_graphs = []
    for i, graph in enumerate(graphs):
        try:
            runnable = graph.run()
            assert isinstance(runnable, Callable)
        except:
            with Session(_engine) as session:
                session.delete(graph)
                session.commit()
            continue
        graph_runs = []
        for j, task_id in enumerate(tasks):
            graph_runs.append(runnable(get_task_data(task_id)))
        run.append(graph_runs)
        valid_graphs.append(graph)
    
    graphs = valid_graphs
    
    assert run, 'all graph corrupted'
    assert len(graphs) == len(run), (graphs, run)
    async def handle_error(x):
        try:
            return await x
        except Exception as e:
            print("ERROR")
            print(e)
            print("END ERROR")
            return ('Error: ' + str(e), False)
    res = [[(await handle_error(x)) for x in y] for y in run]
    for result, graph in zip(res, graphs):
        if all(x[0].startswith('Error') for x in result):
            with Session(_engine) as session:
                session.delete(graph)
                session.commit()
    assert res
    assert res[0]
    assert isinstance(res[0], list)
    score = [sum(x[1] for x in y) / len(y) for y in res]
    assert len(graphs) == len(score)
    assert len(graphs) == len(compute_probabilities(score))
    graph = np.random.choice(graphs, p=compute_probabilities(score))
    return graph


def format_experience(graph):
    failures = [x for x in graph.children if x.score <= graph.score]
    successes = [x for x in graph.children if x.score > graph.score]
    experience = f"Original Score: {graph.score}\n"
    experience += "These are some conclusions drawn from experience:\n\n"
    for failure in failures:
        experience += f"-Absolutely prohibit {failure.modification} (Score: {failure.score})\n"
    for success in successes:
        experience += f"-Absolutely prohibit {success.modification} \n"
    experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."
    return experience

def format_operator_description(operators_doc):
    descriptions = []
    for i, (name, doc) in enumerate(operators_doc.items()):
        descriptions.append(OPERATOR_DESCRIPTION.format(
            id=i,
            name=name,
            description=doc['description'],
            interface=doc['interface']
        ))
    return '\n'.join(descriptions)

def format_log(data):
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() if not isinstance(v, bytes)}
        elif isinstance(d, (list, tuple)):
            return [clean_dict(x) for x in d if not isinstance(x, bytes)]
        elif isinstance(d, Image.Image):
            return f"<Image size={d.size} mode={d.mode}>"
        return d
    log = ""
    for run in data:
        sample = {
            "log": clean_dict(run.log),
            "final_output": run.final_output,
            "task": clean_dict(run.task)  # This should work since get_task_data() returns a dict
        }
        log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"
    return log

class GraphOp(BaseModel):
    thought: str = Field(default="", description="reason and plan for your optimization")
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")

for round in range(max_rounds):
    graph: Graph = asyncio.run(experiment())
    with Session(_engine) as session:
        graph = session.get(Graph, graph.id)
        runs = list(filter(lambda x: not x.correct, graph.runs))
        runs = random.sample(runs, min(3, len(runs)))
        run_imgs = [run.task['image'].thumbnail((764, 764)) for run in runs]

        prompt = WORKFLOW_OPTIMIZE_PROMPT + WORKFLOW_INPUT.format(
            experience = format_experience(graph),
            score = graph.score,
            graph = graph.graph,
            prompt = graph.prompt,
            operator_description = format_operator_description(operators_doc),
            log=format_log(runs)
        )
        
        node = asyncio.run(ActionNode.from_pydantic(GraphOp).fill(
            context=prompt,
            llm=LLM(model='gemini-2.5-pro-exp-03-25'),
            mode="xml_fill",
        ))
        graph_data = node.instruct_content.model_dump()

        new_graph = Graph(
            graph=graph_data['graph'],
            prompt=graph_data['prompt'],
            father_id=graph.id,
            change=graph_data['modification'],
        )
    put(new_graph)

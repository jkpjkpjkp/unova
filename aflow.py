from db import *
import asyncio
from zero import get_task_data
import numpy as np
import json
import random
from pydantic import BaseModel

from aflow_prompt import *
from action_node import ActionNode, LLM

max_rounds = 25

def compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
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

async def experiment(
    num_graph=7,
    num_task=14,
) -> Graph:
    # selecting highest variation is self-stabilizing (good). 
    tasks = get_high_variation_task(num_task)
    graphs = get_strongest_graph(num_graph)

    run = [[None for _ in range(num_task)] for _ in range(num_graph)]
    for i, graph in enumerate(graphs):
        for j, task_id in enumerate(tasks):
            run[i, j] = graph.run(get_task_data(task_id))
    
    res = [[await x for x in y] for y in run]
    score = [sum(x[1] for x in y) / len(y) for y in res]
    graph = graphs[np.random.choice(len(graphs), p=compute_probabilities(score))]
    return graph


def format_experience(graph):
    failures = [x for x in graph.children if x.score <= graph.score]
    successes = [x for x in graph.children if x.score > graph.score]
    experience = f"Original Score: {graph['score']}\n"
    experience += "These are some conclusions drawn from experience:\n\n"
    for failure in failures:
        experience += f"-Absolutely prohibit {failure.modification} (Score: {failure.score})\n"
    for success in successes:
        experience += f"-Absolutely prohibit {success.modification} \n"
    experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."
    return experience

def format_operator_description(operators):
    return '\n'.join(OPERATOR_DESCRIPTION.format(id=i, name=op.name, description=op.description, interface=op.interface) for i, op in enumerate(operators))

def format_log(data):
    log = ""
    for sample in data:
        log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"
    return log

class GraphOp(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")

for round in range(max_rounds):
    graph: Graph = asyncio.run(experiment())
    runs = filter(lambda x: not x.correct, graph.runs)
    runs = random.sample(runs, min(3, len(runs)))
    run_imgs = [run.task[0].thumbnail(764, 764) for run in runs]

    prompt = WORKFLOW_OPTIMIZE_PROMPT + WORKFLOW_INPUT.format(
        experience = format_experience(graph),
        score = graph.score,
        graph = graph.graph,
        prompt = graph.prompt,
        operator_description = format_operator_description(operators),
        log=format_log(runs)
    )
    
    ActionNode.from_pydantic(GraphOp).fill(
        context=prompt,
        llm=LLM(model='gemini-2.5-pro-exp-03-25'),
        mode="xml_fill",
    )
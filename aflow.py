from db import *
import asyncio
from zero import get_task_data
import numpy as np
import json
import os
from collections import defaultdict
import random
from loguru import logger
import itertools
from pydantic import BaseModel

from aflow_prompt import *

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



loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
for round in range(max_rounds):
    graph: Graph = experiment()
    runs = filter(lambda x: not x.correct, graph.runs)
    run = random.choice(runs)
    avoid = [x.change for x in graph.children]

    
    store graph as file, use traditional aflow touch.


    gather experience, log, 
    and make modification.

    sam result will cause massive vlm call, should only SoM and crop. this also make present() Image only. 

    log contain thumbnail of original task image only
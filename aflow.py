from db import *
import asyncio
from zero import get_task_data
import numpy as np
import json
import os
from collections import defaultdict

from loguru import logger
import itertools

def compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
    scores = np.array(scores, dtype=np.float64)
    n = len(scores)

    if n == 0:
        raise ValueError("Score list is empty.")

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

async def experiment():
    tasks = get_high_variation_task(7)
    graphs = get_strongest_graph(7)

    run = [[None for _ in range(7)] for _ in range(7)]
    for i, graph in enumerate(graphs):
        for j, task_id in enumerate(tasks):
            run[i, j] = graph.run(get_task_data(task_id))
    
    res = [[await x for x in y] for y in run]
    score = [sum(x[1] for x in y) / len(y) for y in res]
    graph = graphs[np.random.choice(len(graphs), p=compute_probabilities(score))]


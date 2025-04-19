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


class ExperienceUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def load_experience(self, path=None, mode: str = "Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        experience_data = defaultdict(lambda: {"score": None, "success": {}, "failure": {}})

        for round_dir in os.listdir(rounds_dir):
            if os.path.isdir(os.path.join(rounds_dir, round_dir)) and round_dir.startswith("round_"):
                round_path = os.path.join(rounds_dir, round_dir)
                try:
                    round_number = int(round_dir.split("_")[1])
                    json_file_path = os.path.join(round_path, "experience.json")
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        father_node = data["father node"]

                        if experience_data[father_node]["score"] is None:
                            experience_data[father_node]["score"] = data["before"]

                        if data["succeed"]:
                            experience_data[father_node]["success"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                        else:
                            experience_data[father_node]["failure"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                except Exception as e:
                    logger.info(f"Error processing {round_dir}: {str(e)}")

        experience_data = dict(experience_data)

        output_path = os.path.join(rounds_dir, "processed_experience.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(experience_data, outfile, indent=4, ensure_ascii=False)

        logger.info(f"Processed experience data saved to {output_path}")
        return experience_data

    def format_experience(self, processed_experience, sample_round):
        experience_data = processed_experience.get(sample_round)
        if not experience_data:
            return f"No experience data found for round {sample_round}."
        experience = f"Original Score: {experience_data['score']}\n"
        experience += "These are some conclusions drawn from experience:\n\n"
        for key, value in experience_data["failure"].items():
            experience += f"-Absolutely prohibit {value['modification']} (Score: {value['score']})\n"
        for key, value in experience_data["success"].items():
            experience += f"-Absolutely prohibit {value['modification']} \n"
        experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."
        return experience

    def check_modification(self, processed_experience, modification, sample_round):
        experience_data = processed_experience.get(sample_round)
        if not experience_data:
            return True
        for key, value in itertools.chain(experience_data["failure"].items(), experience_data["success"].items()):
            if value["modification"] == modification:
                return False
        return True

    def create_experience_data(self, sample, modification):
        return {
            "father node": sample["round"],
            "modification": modification,
            "before": sample["score"],
            "after": None,
            "succeed": None,
        }

    def update_experience(self, directory, experience, avg_score):
        experience["after"] = avg_score
        experience["succeed"] = bool(avg_score > experience["before"])

        with open(os.path.join(directory, "experience.json"), "w", encoding="utf-8") as f:
            json.dump(experience, f, indent=4, ensure_ascii=False)

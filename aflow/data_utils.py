import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from loguru import logger

class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []

    def load_results(self, path: str) -> list:
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    def get_top_rounds(self, sample: int, path=None, mode="Graph"):
        self._load_scores(path, mode)
        unique_rounds = set()
        unique_top_scores = []

        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items):
        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
        scores = np.array(scores, dtype=np.float64)
        n = len(scores)
        assert n, "Score list is empty."

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

    def load_log(self, cur_round, path=None, mode: str = "Graph") -> MM:
        """
        Load logs from log.json file, including intermediate execution results.
        This helps the optimizer understand what went wrong during execution.
        
        Returns:
            MM: A MultimodalMessage containing the formatted log data
        """
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, "workflows", f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        if not os.path.exists(log_dir):
            logger.warning(f"No log_dir found for round {cur_round}")
            return ""
        
        logger.info(log_dir)
        data = read_json_file(log_dir, encoding="utf-8")
        
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            logger.warning(f"No log found for round {cur_round}")
            return MM()

        # Sample some failures for analysis, with priority to entries with intermediate results
        final_outputs = [item for item in data if not item.get("is_intermediate", False)]
        intermediate_results = [item for item in data if item.get("is_intermediate", False)]
        
        # Group intermediate results by question
        question_to_steps = {}
        for item in intermediate_results:
            question_text = item["question"]["text"] if isinstance(item["question"], dict) else str(item["question"])
            if question_text not in question_to_steps:
                question_to_steps[question_text] = []
            question_to_steps[question_text].append(item)
        
        # Prioritize questions with intermediate steps
        questions_with_steps = list(question_to_steps.keys())
        
        # Select a few final outputs to show
        sample_size = min(3, len(final_outputs))
        random_samples = []
        
        # First include examples with intermediate steps
        final_with_steps = [
            item for item in final_outputs 
            if isinstance(item["question"], dict) and item["question"]["text"] in questions_with_steps
        ]
        
        if final_with_steps:
            selected_with_steps = random.sample(final_with_steps, min(2, len(final_with_steps)))
            random_samples.extend(selected_with_steps)
        
        # Add more random samples if needed
        remaining_samples_needed = max(0, sample_size - len(random_samples))
        if remaining_samples_needed > 0 and len(final_outputs) > len(random_samples):
            remaining_outputs = [item for item in final_outputs if item not in random_samples]
            random_samples.extend(random.sample(remaining_outputs, min(remaining_samples_needed, len(remaining_outputs))))

        # Format the log with final outputs and their intermediate steps
        log = ""
        for sample in random_samples:
            # First add the final output
            log += "Final Result:\n"
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"
            
            # Then add any intermediate steps
            question_text = sample["question"]["text"] if isinstance(sample["question"], dict) else str(sample["question"])
            if question_text in question_to_steps:
                log += "Intermediate Steps:\n"
                steps = question_to_steps[question_text]
                steps.sort(key=lambda x: x.get("timestamp", ""))
                
                for step in steps:
                    log += f"Step: {step.get('step_name', 'Unknown')}\n"
                    log += json.dumps(step, indent=4, ensure_ascii=False) + "\n\n"
                
                log += "-" * 40 + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float) -> dict:
        now = datetime.datetime.now()
        return {"round": round, "score": score, "avg_cost": avg_cost, "total_cost": total_cost, "time": now}

    def save_results(self, json_file_path: str, data: list):
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def _load_scores(self, path=None, mode="Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = read_json_file(result_file, encoding="utf-8")
        df = pd.DataFrame(data)

        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores

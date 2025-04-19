# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph

import asyncio
import time
import os
from typing import List, Literal

from pydantic import BaseModel, Field
from loguru import logger

from action_node import ActionNode
from aflow.convergence_utils import ConvergenceUtils
from aflow.data_utils import DataUtils
from aflow.evaluation_utils import EvaluationUtils
from aflow.experience_utils import ExperienceUtils
from aflow.graph_utils import GraphUtils
from action_node import LLM

QuestionType = Literal["math", "code", "qa", "vqa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: str,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 1,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = LLM(model='gemini-2.5-pro-exp-03-25')
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path, dataset)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 3  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            score = loop.run_until_complete(self._optimize_graph())
            
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True)

        while True:
            directory = os.path.join(graph_path, f"round_{self.round + 1}")
            os.makedirs(directory, exist_ok=True)
            top_rounds = self.data_utils.get_top_rounds(self.sample)

            sample = self.data_utils.select_round(top_rounds)
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])
            operator_description = self.graph_utils.load_operators_description(self.operators)
            log_data = self.data_utils.load_log(sample["round"])

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
                )
            graph_optimize_node = await ActionNode.from_pydantic(GraphOptimize).fill(
                req=graph_optimize_prompt, mode="xml_fill", llm=self.optimize_llm
                )
            response = await self.graph_utils.get_graph_optimize_response(graph_optimize_node)

            if not self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
                ):
                continue
            break

        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)
        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])
        logger.info(directory)
        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False)
        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    async def test(self):
        rounds = [5]
        data = []

        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True)

            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)

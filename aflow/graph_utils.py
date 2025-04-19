import json
import os
import re
import time
import traceback
from typing import List

from metagpt.ext.aflow.scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
)
from loguru import logger

class GraphUtils:
    def __init__(self, root_path: str, dataset: str):
        self.root_path = root_path
        self.dataset = dataset
    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def load_prompt_custom(self, round_number: int):

    # llm_config = ModelsConfig.default().get(llm_name)
        class Prompts:
            pass

        def import_symbols_to_class(module_name, target_class):
            module = __import__(module_name, fromlist=[""])
            
            symbols = getattr(module, '__all__', None)
            if symbols is None:
                symbols = [name for name in dir(module) if not name.startswith('_')]
            
            for symbol in symbols:
                symbol_value = getattr(module, symbol)
                setattr(target_class, symbol, symbol_value)

        import_symbols_to_class(
            f"metagpt.ext.aflow.scripts.optimized.{self.dataset}.workflows.round_{round_number}.prompt", Prompts
            )
        return Prompts()

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        if workflows_path.endswith("."):
            workflows_path = workflows_path[:-1]
        
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            Graph = getattr(graph_module, "Graph")
            graph = Graph(operators=operators_dict, prompt_custom=self.load_prompt_custom(round_number))
            return graph.run
        except ImportError as e:
            logger.info(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.info(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        """Extract class Graph
        
        Explanation:
            .*? non-greedily matches any character (including newlines due to re.DOTALL).
            (?=^\S|\Z) is a lookahead that stops the match at the start of a line with a non-whitespace character (^\S) or the end of the string (\Z).
            re.MULTILINE allows ^ to match the start of any line.
            re.DOTALL lets . include newlines, ensuring the entire class body is matched.
        """
        pattern = re.compile(r"class Graph:.*?(?=^\S|\Z)", re.DOTALL | re.MULTILINE)
        return re.findall(pattern, graph_load)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operators_description += self._load_operator_description(id + 1, operator, path)
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface}).\n"

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: MM,
    ) -> str:
        with open("metagpt/ext/aflow/scripts/message.py", "r") as file:
            message_py = file.read()
        graph_input = powerformat(
            WORKFLOW_INPUT,
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            multimodal_message_description=message_py,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.info(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None
    
    def remove_3_ticks(self, text: str) -> str:
        if text[:9] == "```python" and text[-3:] == "```":
            return text[9:-3]
        return text

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        graph=self.remove_3_ticks(response["graph"])
        prompt=self.remove_3_ticks(response["prompt"])
        graph = WORKFLOW_TEMPLATE.format(graph=graph, round=round_number, dataset=dataset)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(prompt)

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")

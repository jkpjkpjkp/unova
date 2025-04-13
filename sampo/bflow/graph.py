import asyncio
from experiment_store import Graph, Run
import re

class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom
    async def run(self, run: list[Run]) -> Graph:
        run = run[0]
        prompt = self.prompt_custom['AFLOW'].format(
            type = run.graph.task_tag,
            graph = run.graph.graph,
            prompt = run.graph.prompt,
            operator_description = r""""Custom": {
        "description": "A basic operator that executes custom instructions on input",
        "interface": "async def __call__(self, input, instruction)"
    },""",
            log = run.log,
        )
        response = await self.custom(input=prompt, instruction="")
        return Graph(
            graph = re.match(r"<graph>(.*?)</graph>", response, re.DOTALL).group(1),
            prompt = re.match(r"<prompt>(.*?)</prompt>", response, re.DOTALL).group(1),
            task_tag = run.graph.task_tag,
        )
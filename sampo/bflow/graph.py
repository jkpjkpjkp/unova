import asyncio
from experiment_store import Graph, Run
import re
from anode import xml_extract

class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.operators = operators
        self.prompt_custom = prompt_custom
    async def run(self, run: list[Run]) -> Graph:
        assert len(run) == 1
        run = run[0]
        prompt = self.prompt_custom['AFLOW'].format(
            type = run.task_tag,
            graph = run.graph.graph,
            prompt = run.graph.prompt,
            operator_description = self.operator_descriptions,
            log = run.log,
        )
        response = await self.custom(input=prompt)
        data = xml_extract(response, ['graph', 'prompt'], {'graph': str, 'prompt': str})
        return Graph(
            graph = data['graph'],
            prompt = data['prompt'],
            task_tag = run.task_tag,
        )
import asyncio
from db import Graph as Graph_, Run
import re
import asyncio
from anode import xml_extract
import inspect
from typing import Callable

class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom
    
    async def run(self, run: list[Run]) -> Graph_:
        assert len(run) == 1
        run = run[0]
        prompt = self.prompt_custom['CORRECT_IT'].format(
            question = run.task.task,
            answer = run.task.answer,
            graph = run.graph.graph,
            prompt = run.graph.prompt,
            log = run.log,
        )
        prompt += self.prompt_custom['CUSTOM_USE']
        response = await self.custom(input=prompt)
        data = xml_extract(response, ['graph', 'prompt'], {'graph': str, 'prompt': str})
        return Graph_(
            graph = data['graph'],
            prompt = data['prompt'],
        )
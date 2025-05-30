from db import Graph as Graph_, Run
from anode import xml_extract

class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.prompts = prompts
    
    async def run(self, run: list[Run]) -> Graph_:
        assert len(run) == 1
        run = run[0]
        prompt = self.prompts['CORRECT_IT'].format(
            question = run.task.task,
            answer = run.task.answer,
            graph = run.graph.graph,
            prompt = run.graph.prompt,
            log = run.log,
        )
        prompt += self.prompts['CUSTOM_USE']
        response = await self.custom(input=prompt)
        data = xml_extract(response, ['graph', 'prompt'], {'graph': str, 'prompt': str})
        return Graph_(
            graph = data['graph'],
            prompt = data['prompt'],
        )
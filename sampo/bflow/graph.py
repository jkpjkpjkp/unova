from db import Graph as Graph_, Run
from anode import xml_hint, xml_extract
import inspect
from typing import Callable
class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.operators = operators
        self.prompt_custom = prompt_custom

    def _generate_operator_descriptions(self) -> str:
        descriptions = []
        for name, operator in self.operators.items():
            if isinstance(operator, Callable):
                sig = inspect.signature(operator)
                doc = inspect.getdoc(operator)
                descriptions.append(f"Operator: {name}\nSignature: {sig}\n{'Docstring:\n' + doc if doc else ''}\n")
        return "\n".join(descriptions)
    
    def _format_experience(self, experience: list[tuple[str, float]]) -> str:
        return "\n".join([f"-Absolutely prohibit {modification}\n (Score: {score})" for modification, score in experience])

    async def run(self, run: list[Run]) -> Graph_:
        assert len(run) == 1
        run = run[0]
        prompt = self.prompt_custom['AFLOW'].format(
            type = 'vqa',
            # experience = self._format_experience(run.experience),
            graph = run.graph.graph,
            prompt = run.graph.prompt,
            score = run.graph.score,
            operator_description = self._generate_operator_descriptions(),
            log = run.log,
        )
        prompt += self.prompt_custom['CUSTOM_USE']
        prompt += xml_hint(['graph', 'prompt'])
        response = await self.custom(input=prompt, model='gemini-2.5-pro-exp-03-25')
        data = xml_extract(response, ['graph', 'prompt'], {'graph': str, 'prompt': str})
        return Graph_(
            graph = data['graph'],
            prompt = data['prompt'],
        )
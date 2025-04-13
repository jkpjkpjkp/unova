import asyncio
from experiment_store import Graph, Run
import re

class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom
    
    async def run(self, run: list[Run]) -> Graph:
        graph1 = run[0].graph
        graph2 = run[1].graph
        task1 = run[0].task
        task2 = run[2].task
        assert graph1 != graph2
        assert task1 != task2
        assert run[0].correct and not run[1].correct
        assert not run[2].correct and run[3].correct
        prompt = self.prompt_custom['MERGE_PROMPT'].format(graph1=graph1, graph2=graph2, task1=task1, task2=task2, graph1_on_task_1_correct=run[0], graph2_on_task_1_wrong=run[1], graph1_on_task_2_wrong=run[2], graph2_on_task_2_correct=run[3])
        response = await self.custom(input=prompt, instruction="")
        return Graph(
            graph = re.match(r"<graph>(.*?)</graph>", response, re.DOTALL).group(1),
            prompt = re.match(r"<prompt>(.*?)</prompt>", response, re.DOTALL).group(1),
            task_tag = run.task_tag,
        )
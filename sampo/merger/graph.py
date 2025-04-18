from db import Graph as Graph_, Run
from anode import xml_extract

class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom
    
    async def run(self, run: list[Run]) -> Graph_:
        graph1 = run[0].graph
        graph2 = run[1].graph
        prompt = self.prompt_custom['MP'].format(graph1=graph1.graph, prompt1=graph1.prompt, graph2=graph2.graph, prompt2=graph2.prompt)
        prompt += self.prompt_custom['CUSTOM_USE']
        response = await self.custom(input=prompt)
        field_names = ['graph', 'prompt']
        field_types = {'graph': str, 'prompt': str}
        data = xml_extract(response, field_names, field_types)
        return Graph_(graph=data['graph'], prompt=data['prompt'])
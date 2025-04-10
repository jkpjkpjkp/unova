
class Graph:
    def __init__(self, operators: dict, prompt_custom: object):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom

    async def run(self, problem: str) -> str:
        return await self.custom(input=problem, instruction=self.prompt_custom.COT_PROMPT)

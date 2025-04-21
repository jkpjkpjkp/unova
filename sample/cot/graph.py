class Graph:
    def __init__(self, operators: dict, prompts: dict):
        self.custom = operators['Custom']
        self.prompts = prompts

    async def run(self, problem: str) -> str:
        return (await self.custom(input=problem + self.prompts['COT_PROMPT']))['response']

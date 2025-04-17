class Graph:
    def __init__(self, operators: dict, prompt_custom: dict):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom

    async def run(self, problem: str) -> str:
        return await self.custom(input=problem + self.prompt_custom['COT'], model='gemini-2.5-pro-exp-03-25')

from action_node import LLM, ActionNode

from pydantic import BaseModel, Field
from action_node import ActionNode


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class BboxOp(BaseModel):
    thought: str = Field(default="", description="Thoughts on what crop may be most useful.")
    bbox: tuple[int, int, int, int] = Field(default=[], description="a crop containing all relevant information, in x y x y format, idx from 0 to 1000")

class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"req": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)

        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

class Custom(Operator):
    def __init__(self, llm: LLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, pydantic_model=GenerateOp, mode: Literal["single_fill", "xml_fill", "code_fill"] = "single_fill"):
        prompt = input
        response = await self._fill_node(pydantic_model, prompt, mode=mode)
        return response

class Bbox(Operator):
    def __init__(self, llm: LLM, name: str = "Bbox"):
        super().__init__(llm, name)

    async def __call__
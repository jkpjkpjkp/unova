MP = """Here are 2 workflows with different strengths. try to combine and merge them into one which can solve all problems solvable by the two.

<graph1>
    <graph>{graph1}</graph>
    <prompt>{prompt1}</prompt>
</graph1>

<graph2>
    <graph>{graph2}</graph>
    <prompt>{prompt2}</prompt>
</graph2>

you should follow this xml format and split your graph class and prompt definition code.
"""

from sampo.aflow.prompt import CUSTOM_USE

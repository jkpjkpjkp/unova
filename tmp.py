import re

def extract_answer(s):
    start = s.find("\\boxed{")
    if start == -1:
        return None

    idx = start + len("\\boxed{")
    brace_level = 1

    answer = ""
    while idx < len(s) and brace_level > 0:
        c = s[idx]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        answer += c
        idx += 1

    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\!", "", answer)
    return answer.strip()

if __name__ == "__main__":
    output = "The answer is \\boxed{123}."
    print(extract_answer(output))


from pydantic import BaseModel

def xml_call(input: str, output: BaseModel) -> dict:
    pass





MERGE_PROMPT = 'We are optimizing llm workflows. Below are 2 workflows and 2 tasks, each workflow is correct on exactly one task. I want you to analyze why for each task one got it right and the other wrong. After the analysis I desire you to come up with a third workflow combining the strengths of the two, that will be correct on both tests. This workflow should be written with Integrity, that is, it will not give away specifics of the two tests it will face.'
MERGE_PROMPT_TAIL = '<graph1>{graph1}</graph1><graph2>{graph2}</graph2><task1>{task1}</task1><task2>{task2}</task2><graph1_on_task_1(correct)>{graph1_on_task_1_correct}</graph1_on_task_1(correct)><graph2_on_task_1(wrong)>{graph2_on_task_1}</graph2_on_task_1(wrong)><graph1_on_task_2(wrong)>{graph1_on_task_2}</graph1_on_task_2(wrong)><graph2_on_task_2(correct)>{graph2_on_task_2}</graph2_on_task_2(correct)>'



##### UNDONE BELOW


"""
# wrong traj
Here is a workflow in the form of python code and it got wrong on this particular task. please see his trajectory and the ground truth final answer and try to deduce which Step went wrong after so please refine the workflow so as to reach the correct answer of course you should not give the answer away or use any method that will not be generally applicable to other problems similar.


# concisify
Here is a workflow in the form of python code that correctly answered a task however since some of the steps or prompts may be redundant or suboptimal you are tasked to reduce this workflow to a more concise form that can still correctly solve this task. 


#hard task
for extremely hard tasks measures must be taken such that

0. it speculates which step is incorret
1. llm guesses its trail to correct answer.
2. ()
"""


import asyncio

class Graph:
    def __init__(self, operators: dict, prompt_custom: object):
        self.custom = operators['Custom']
        # Crop operator is available but not used in this version
        # self.crop = operators['Crop']
        self.prompt_custom = prompt_custom # Contains LOCATE_PROMPT, VQA_PROMPT, REFINE_VQA_PROMPT

    async def run(self, problem: str) -> str:
        """
        Answers the visual question using a locate-VQA-refine pipeline.
        1. Identify relevant area/objects.
        2. Perform initial VQA focused on that area.
        3. Refine the VQA answer, considering estimation for capacity questions.
        """
        # Step 1: Identify relevant area/objects
        locate_instruction = self.prompt_custom['LOCATE_PROMPT']
        location_description = await self.custom(input=problem, instruction=locate_instruction)

        # Step 2: Perform Initial VQA focused on the identified area
        vqa_instruction = self.prompt_custom['VQA_PROMPT']
        vqa_input = problem
        focus_desc_text = ""
        if str(location_description).strip() and "Could not locate specific area." not in str(location_description):
             focus_desc_text = "\nFocus Area Description:\n" + str(location_description)
             vqa_input += str(focus_desc_text)

        initial_answer = await self.custom(input=vqa_input, instruction=vqa_instruction)

        # Step 3: Refine VQA answer, especially for estimation
        refine_instruction = self.prompt_custom['REFINE_VQA_PROMPT']
        # Combine original problem, focus description (if any), and initial answer for refinement context
        refine_input = problem + str(focus_desc_text) + str("\n\nInitial Answer:\n") + initial_answer
        final_answer = await self.custom(input=refine_input, instruction=refine_instruction)

        return final_answer



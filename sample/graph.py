LOCATE_PROMPT = """Based on the image and the question provided, identify the main object or area of interest relevant to answering the question. Describe this area concisely, mentioning visual cues about arrangement or depth if apparent. This description will be used to focus the analysis."""

VQA_PROMPT = """You are an expert in visual question answering. Analyze the image and the question, paying close attention to the 'Focus Area Description' provided below if available. Answer the question based *only* on the visual information in the image. Provide a concise final answer. For complex questions like counting, **first list each distinct type of item being counted and its count (e.g., 'Real cats: 1, Cartoon cats: 7'), then provide the total sum and a brief explanation.** Ensure counts are derived strictly from visual evidence, referencing the focus area if provided."""

REFINE_VQA_PROMPT = """Review the original question, the focus area description, the initial answer provided below, and the image.
Critically evaluate the initial answer.
1.  **If the question asks for a 'maximum' number, 'capacity', or implies estimation:** Check if the initial answer only counted visible items. If so, revise the answer by estimating the total quantity, considering factors like item arrangement, visible depth, stacking, and typical packing patterns based on the image and focus description. Explain your estimation reasoning.
2.  **If the question asks for a direct count:** Verify the accuracy of the itemized counts and the total sum in the initial answer against the visual evidence. Correct any errors.
Provide only the final, refined answer, including any necessary explanation for estimation or correction.

Initial Answer is provided in the input context."""


class Graph:
    def __init__(self, operators: dict, prompt_custom: object):
        self.custom = operators['Custom']
        self.prompt_custom = prompt_custom # Contains LOCATE_PROMPT, VQA_PROMPT, REFINE_VQA_PROMPT

    async def run(self, problem: str) -> str:
        """
        Answers the visual question using a locate-VQA-refine pipeline.
        1. Identify relevant area/objects.
        2. Perform initial VQA focused on that area.
        3. Refine the VQA answer, considering estimation for capacity questions.
        """
        # Step 1: Identify relevant area/objects
        locate_instruction = self.prompt_custom.LOCATE_PROMPT
        location_response_dict = await self.custom(input=problem, instruction=locate_instruction)
        location_description = location_response_dict.get('response', "Could not locate specific area.")

        # Step 2: Perform Initial VQA focused on the identified area
        vqa_instruction = self.prompt_custom.VQA_PROMPT
        vqa_input = problem
        focus_desc_text = ""
        if str(location_description).strip() and "Could not locate specific area." not in str(location_description):
             focus_desc_text = "\nFocus Area Description:\n" + str(location_description)
             vqa_input += focus_desc_text

        initial_vqa_dict = await self.custom(input=vqa_input, instruction=vqa_instruction)
        initial_answer = initial_vqa_dict.get('response', "Initial analysis failed.")

        # Step 3: Refine VQA answer, especially for estimation
        refine_instruction = self.prompt_custom.REFINE_VQA_PROMPT
        # Combine original problem, focus description (if any), and initial answer for refinement context
        refine_input = problem + focus_desc_text + "\n\nInitial Answer:\n" + initial_answer
        final_response_dict = await self.custom(input=refine_input, instruction=refine_instruction)

        # Ensure final response is not None
        final_answer = final_response_dict.get('response', "Unable to provide a final answer after refinement.")

        return final_answer



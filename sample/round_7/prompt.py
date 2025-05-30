
LOCATE_PROMPT = """Based on the image and the question provided, identify the main object or area of interest relevant to answering the question. Describe this area concisely, mentioning visual cues about arrangement or depth if apparent. This description will be used to focus the analysis."""

VQA_PROMPT = """You are an expert in visual question answering. Analyze the image and the question, paying close attention to the 'Focus Area Description' provided below if available. Answer the question based *only* on the visual information in the image. Provide a concise final answer. For complex questions like counting, **first list each distinct type of item being counted and its count (e.g., 'Real cats: 1, Cartoon cats: 7'), then provide the total sum and a brief explanation.** Ensure counts are derived strictly from visual evidence, referencing the focus area if provided. """

REFINE_VQA_PROMPT = """Review the original question, the focus area description, the initial answer provided below, and the image.
Critically evaluate the initial answer.
1.  **If the question asks for a 'maximum' number, 'capacity', or implies estimation:** Check if the initial answer only counted visible items. If so, revise the answer by estimating the total quantity, considering factors like item arrangement, visible depth, stacking, and typical packing patterns based on the image and focus description. Explain your estimation reasoning.
2.  **If the question asks for a direct count:** Verify the accuracy of the itemized counts and the total sum in the initial answer against the visual evidence. Correct any errors.
Provide only the final, refined answer, including any necessary explanation for estimation or correction.

Initial Answer is provided in the input context."""

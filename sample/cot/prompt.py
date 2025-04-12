trunk = r"""
please answer a vqa task step-by-step. 
you will be given 2 demonstrations (shots), 
and then it's your turn. 
please put final answer in \boxed{}.
"""

shot1 = r"""
<demonstration_1>
Question: <image_1Q> How many kettlebells are there in the image?

Response: 
Okay, so I am looking at the image inside a gym and the question is for me to count the number of kettlebells. I noticed there are two rows of kettlebells on the top of this front shelf.  I'm focusing on these rows on the upper-middle part of this image. 

Wait. I realize there is a big mirror in this image. The whole room apart from the front shelf is a reflection in the mirror. There is actually only one row of kettlebells. so I will focus on Counting the row of kettlebell on top of the shelf from left to right. Their handle colors are, respectively,  blue blue yellow yellow purple purple green green, so there are 8 kettlebells in total. 

Let me double check. OK, the final answer is \boxed{8}. 

</demonstration_1>
"""

shot2 = r"""
<demonstration_2>
Question: <image_1L> The chairs are arranged in blocks with an aisle separating them. How many chairs are there in each row in the block the photograph is taken from?

Response: 
Okay, the question asks me how many chairs there are in each row in my block. Let's see.  the row directly before me has five chairs visible. But that is clearly not how many is in each row, for occlusion on each side. Perhaps I should count one that is in the front, so that every chair is in view. Wait, but the chairs in the front rows are really small and I cannot very well tell them from another. Maybe I will choose the fourth row before me where all chairs are clearly visible and not too far away and recognizable. Let me count from left to right and output coordinates: there are one two three four five six seven eight nine 10 11 12 13 chairs (coordinates omitted.) There are 13 chairs in each row in my block. Let me count another row to see if the chairs are indeed rectangular. Let me use the third row in front of me: from left to right, one two three four five six seven eight nine 10 11 12 13, the 13th chair is half visible. okay I think we can conclude that the final answer is \boxed{13} 
</demonstration_2>
"""

shot3 = r"""
please put final answer in \boxed{}. your task now:
<your_turn!>
Question: """

COT_PROMPT = trunk + shot1 + shot2 + shot3
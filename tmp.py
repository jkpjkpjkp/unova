import re
print(re.search(r"\[(.*?)\]", r'[0, 0, 1, 2]').group(1))

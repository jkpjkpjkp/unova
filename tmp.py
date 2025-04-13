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


import json
import re
import typing
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Type, Union, Literal
from pydantic import BaseModel, Field, create_model, model_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from loguru import logger
import base64
from io import BytesIO
import copy
from typing import Callable
from tenacity import RetryCallState, wait_fixed
import loguru
from json import JSONDecodeError
from json.decoder import _decode_uXXXX
import asyncio
from PIL import Image
from pydantic import BeforeValidator
from typing_extensions import Annotated
import uuid

class LLM:
    def __init__(self, model='gemini-2.0-flash') -> None:
        self.model = model
        self.client = OpenAI(
            api_key='sk-local',
            base_url='http://localhost:7912'
        )
    async def aask(self, prompt, system_msgs='', images=[], timeout=10):
        messages = [
            {"role": "system", "content": system_msgs or "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        if images:

            def to_base64(image: Image.Image):
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue())
            image_content = [{"type": "image_url", "image_url": {"url": to_base64(img).decode()}} for img in images]
            if isinstance(messages[1]["content"], list):
                messages[1]["content"].extend(image_content)
            else: # Should not happen based on current structure, but good practice
                messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]}, *image_content]

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        ).choices[0].message.content


NUMBER_RE = re.compile(r"(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?", (re.VERBOSE | re.MULTILINE | re.DOTALL))


def py_make_scanner(context):
    parse_object = context.parse_object
    parse_array = context.parse_array
    parse_string = context.parse_string
    match_number = NUMBER_RE.match
    strict = context.strict
    parse_float = context.parse_float
    parse_int = context.parse_int
    parse_constant = context.parse_constant
    object_hook = context.object_hook
    object_pairs_hook = context.object_pairs_hook
    memo = context.memo

    def _scan_once(string, idx):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar in ("'", '"'):
            if idx + 2 < len(string) and string[idx + 1] == nextchar and string[idx + 2] == nextchar:
                # Handle the case where the next two characters are the same as nextchar
                return parse_string(string, idx + 3, strict, delimiter=nextchar * 3)  # triple quote
            else:
                # Handle the case where the next two characters are not the same as nextchar
                return parse_string(string, idx + 1, strict, delimiter=nextchar)
        elif nextchar == "{":
            return parse_object((string, idx + 1), strict, _scan_once, object_hook, object_pairs_hook, memo)
        elif nextchar == "[":
            return parse_array((string, idx + 1), _scan_once)
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            return None, idx + 4
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            return True, idx + 4
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            return False, idx + 5

        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return res, m.end()
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return parse_constant("NaN"), idx + 3
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return parse_constant("Infinity"), idx + 8
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return parse_constant("-Infinity"), idx + 9
        else:
            raise StopIteration(idx)

    def scan_once(string, idx):
        try:
            return _scan_once(string, idx)
        finally:
            memo.clear()

    return scan_once


FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
STRINGCHUNK = re.compile(r'(.*?)(["\\\x00-\x1f])', FLAGS)
STRINGCHUNK_SINGLEQUOTE = re.compile(r"(.*?)([\'\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_DOUBLE_QUOTE = re.compile(r"(.*?)(\"\"\"|[\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_SINGLEQUOTE = re.compile(r"(.*?)('''|[\\\x00-\x1f])", FLAGS)
BACKSLASH = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
WHITESPACE = re.compile(r"[ \t\n\r]*", FLAGS)
WHITESPACE_STR = " \t\n\r"


def JSONObject(
    s_and_end, strict, scan_once, object_hook, object_pairs_hook, memo=None, _w=WHITESPACE.match, _ws=WHITESPACE_STR
):
    """Parse a JSON object from a string and return the parsed object.

    Args:
        s_and_end (tuple): A tuple containing the input string to parse and the current index within the string.
        strict (bool): If `True`, enforces strict JSON string decoding rules.
            If `False`, allows literal control characters in the string. Defaults to `True`.
        scan_once (callable): A function to scan and parse JSON values from the input string.
        object_hook (callable): A function that, if specified, will be called with the parsed object as a dictionary.
        object_pairs_hook (callable): A function that, if specified, will be called with the parsed object as a list of pairs.
        memo (dict, optional): A dictionary used to memoize string keys for optimization. Defaults to None.
        _w (function): A regular expression matching function for whitespace. Defaults to WHITESPACE.match.
        _ws (str): A string containing whitespace characters. Defaults to WHITESPACE_STR.

    Returns:
        tuple or dict: A tuple containing the parsed object and the index of the character in the input string
        after the end of the object.
    """

    s, end = s_and_end
    pairs = []
    pairs_append = pairs.append
    # Backwards compatibility
    if memo is None:
        memo = {}
    memo_get = memo.setdefault
    # Use a slice to prevent IndexError from being raised, the following
    # check will raise a more specific ValueError if the string is empty
    nextchar = s[end : end + 1]
    # Normally we expect nextchar == '"'
    if nextchar != '"' and nextchar != "'":
        if nextchar in _ws:
            end = _w(s, end).end()
            nextchar = s[end : end + 1]
        # Trivial empty object
        if nextchar == "}":
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                return result, end + 1
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return pairs, end + 1
        elif nextchar != '"':
            raise JSONDecodeError("Expecting property name enclosed in double quotes", s, end)
    end += 1
    while True:
        if end + 1 < len(s) and s[end] == nextchar and s[end + 1] == nextchar:
            # Handle the case where the next two characters are the same as nextchar
            key, end = scanstring(s, end + 2, strict, delimiter=nextchar * 3)
        else:
            # Handle the case where the next two characters are not the same as nextchar
            key, end = scanstring(s, end, strict, delimiter=nextchar)
        key = memo_get(key, key)
        # To skip some function call overhead we optimize the fast paths where
        # the JSON key separator is ": " or just ":".
        if s[end : end + 1] != ":":
            end = _w(s, end).end()
            if s[end : end + 1] != ":":
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        end += 1

        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass

        try:
            value, end = scan_once(s, end)
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        pairs_append((key, value))
        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _w(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ""
        end += 1

        if nextchar == "}":
            break
        elif nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        end = _w(s, end).end()
        nextchar = s[end : end + 1]
        end += 1
        if nextchar != '"':
            raise JSONDecodeError("Expecting property name enclosed in double quotes", s, end - 1)
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return result, end
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return pairs, end


def py_scanstring(s, end, strict=True, _b=BACKSLASH, _m=STRINGCHUNK.match, delimiter='"'):
    """Scan the string s for a JSON string.

    Args:
        s (str): The input string to be scanned for a JSON string.
        end (int): The index of the character in `s` after the quote that started the JSON string.
        strict (bool): If `True`, enforces strict JSON string decoding rules.
            If `False`, allows literal control characters in the string. Defaults to `True`.
        _b (dict): A dictionary containing escape sequence mappings.
        _m (function): A regular expression matching function for string chunks.
        delimiter (str): The string delimiter used to define the start and end of the JSON string.
            Can be one of: '"', "'", '\"""', or "'''". Defaults to '"'.

    Returns:
        tuple: A tuple containing the decoded string and the index of the character in `s`
        after the end quote.
    """

    chunks = []
    _append = chunks.append
    begin = end - 1
    if delimiter == '"':
        _m = STRINGCHUNK.match
    elif delimiter == "'":
        _m = STRINGCHUNK_SINGLEQUOTE.match
    elif delimiter == '"""':
        _m = STRINGCHUNK_TRIPLE_DOUBLE_QUOTE.match
    else:
        _m = STRINGCHUNK_TRIPLE_SINGLEQUOTE.match
    while 1:
        chunk = _m(s, end)
        if chunk is None:
            raise JSONDecodeError("Unterminated string starting at", s, begin)
        end = chunk.end()
        content, terminator = chunk.groups()
        # Content is contains zero or more unescaped string characters
        if content:
            _append(content)
        # Terminator is the end of string, a literal control character,
        # or a backslash denoting that an escape sequence follows
        if terminator == delimiter:
            break
        elif terminator != "\\":
            if strict:
                # msg = "Invalid control character %r at" % (terminator,)
                msg = "Invalid control character {0!r} at".format(terminator)
                raise JSONDecodeError(msg, s, end)
            else:
                _append(terminator)
                continue
        try:
            esc = s[end]
        except IndexError:
            raise JSONDecodeError("Unterminated string starting at", s, begin) from None
        # If not a unicode escape sequence, must be in the lookup table
        if esc != "u":
            try:
                char = _b[esc]
            except KeyError:
                msg = "Invalid \\escape: {0!r}".format(esc)
                raise JSONDecodeError(msg, s, end)
            end += 1
        else:
            uni = _decode_uXXXX(s, end)
            end += 5
            if 0xD800 <= uni <= 0xDBFF and s[end : end + 2] == "\\u":
                uni2 = _decode_uXXXX(s, end + 1)
                if 0xDC00 <= uni2 <= 0xDFFF:
                    uni = 0x10000 + (((uni - 0xD800) << 10) | (uni2 - 0xDC00))
                    end += 6
            char = chr(uni)
        _append(char)
    return "".join(chunks), end


scanstring = py_scanstring


class CustomDecoder(json.JSONDecoder):
    def __init__(
        self,
        *,
        object_hook=None,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        strict=True,
        object_pairs_hook=None
    ):
        super().__init__(
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )
        self.parse_object = JSONObject
        self.parse_string = py_scanstring
        self.scan_once = py_make_scanner(self)

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        return super().decode(s)


    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        return super().decode(s)



class RepairType(Enum):
    CS = "case sensitivity"
    RKPM = "required key pair missing"  # condition like `[key] xx` which lacks `[/key]`
    SCM = "special character missing"  # Usually the req_key appear in pairs like `[key] xx [/key]`
    JSON = "json format"


def repair_case_sensitivity(output: str, req_key: str) -> str:
    """
    usually, req_key is the key name of expected json or markdown content, it won't appear in the value part.
    fix target string `"Shared Knowledge": ""` but `"Shared knowledge": ""` actually
    """
    if req_key in output:
        return output

    output_lower = output.lower()
    req_key_lower = req_key.lower()
    if req_key_lower in output_lower:
        # find the sub-part index, and replace it with raw req_key
        lidx = output_lower.find(req_key_lower)
        source = output[lidx : lidx + len(req_key_lower)]
        output = output.replace(source, req_key)
        logger.info(f"repair_case_sensitivity: {req_key}")

    return output


def repair_special_character_missing(output: str, req_key: str = "[/CONTENT]") -> str:
    """
    fix
        1. target string `[CONTENT] xx [CONTENT] xxx [CONTENT]` lacks `/` in the last `[CONTENT]`
        2. target string `xx [CONTENT] xxx [CONTENT] xxxx` lacks `/` in the last `[CONTENT]`
    """
    sc_arr = ["/"]

    if req_key in output:
        return output

    for sc in sc_arr:
        req_key_pure = req_key.replace(sc, "")
        appear_cnt = output.count(req_key_pure)
        if req_key_pure in output and appear_cnt > 1:
            # req_key with special_character usually in the tail side
            ridx = output.rfind(req_key_pure)
            output = f"{output[:ridx]}{req_key}{output[ridx + len(req_key_pure):]}"
            logger.info(f"repair_special_character_missing: {sc} in {req_key_pure} as position {ridx}")

    return output


def repair_required_key_pair_missing(output: str, req_key: str = "[/CONTENT]") -> str:
    """
    implement the req_key pair in the begin or end of the content
        req_key format
            1. `[req_key]`, and its pair `[/req_key]`
            2. `[/req_key]`, and its pair `[req_key]`
    """
    sc = "/"  # special char
    if req_key.startswith("[") and req_key.endswith("]"):
        if sc in req_key:
            left_key = req_key.replace(sc, "")  # `[/req_key]` -> `[req_key]`
            right_key = req_key
        else:
            left_key = req_key
            right_key = f"{req_key[0]}{sc}{req_key[1:]}"  # `[req_key]` -> `[/req_key]`

        if left_key not in output:
            output = left_key + "\n" + output
        if right_key not in output:

            def judge_potential_json(routput: str, left_key: str) -> Union[str, None]:
                ridx = routput.rfind(left_key)
                if ridx < 0:
                    return None
                sub_output = routput[ridx:]
                idx1 = sub_output.rfind("}")
                idx2 = sub_output.rindex("]")
                idx = idx1 if idx1 >= idx2 else idx2
                sub_output = sub_output[: idx + 1]
                return sub_output

            if output.strip().endswith("}") or (output.strip().endswith("]") and not output.strip().endswith(left_key)):
                # # avoid [req_key]xx[req_key] case to append [/req_key]
                output = output + "\n" + right_key
            elif judge_potential_json(output, left_key) and (not output.strip().endswith(left_key)):
                sub_content = judge_potential_json(output, left_key)
                output = sub_content + "\n" + right_key

    return output


def repair_json_format(output: str) -> str:
    """
    fix extra `[` or `}` in the end
    """
    output = output.strip()

    if output.startswith("[{"):
        output = output[1:]
        logger.info(f"repair_json_format: {'[{'}")
    elif output.endswith("}]"):
        output = output[:-1]
        logger.info(f"repair_json_format: {'}]'}")
    elif output.startswith("{") and output.endswith("]"):
        output = output[:-1] + "}"

    # remove comments in output json string, after json value content, maybe start with #, maybe start with //
    arr = output.split("\n")
    new_arr = []
    for json_line in arr:
        # look for # or // comments and make sure they are not inside the string value
        comment_index = -1
        for match in re.finditer(r"(\".*?\"|\'.*?\')|(#|//)", json_line):
            if match.group(1):  # if the string value
                continue
            if match.group(2):  # if comments
                comment_index = match.start(2)
                break
        # if comments, then delete them
        if comment_index != -1:
            json_line = json_line[:comment_index].rstrip()
        new_arr.append(json_line)
    output = "\n".join(new_arr)
    return output


def _repair_llm_raw_output(output: str, req_key: str, repair_type: RepairType = None) -> str:
    repair_types = [repair_type] if repair_type else [item for item in RepairType if item not in [RepairType.JSON]]
    for repair_type in repair_types:
        if repair_type == RepairType.CS:
            output = repair_case_sensitivity(output, req_key)
        elif repair_type == RepairType.RKPM:
            output = repair_required_key_pair_missing(output, req_key)
        elif repair_type == RepairType.SCM:
            output = repair_special_character_missing(output, req_key)
        elif repair_type == RepairType.JSON:
            output = repair_json_format(output)
    return output


def repair_llm_raw_output(
    output: str, req_keys: list[str], repair_type: RepairType = None,
) -> str:
    """
    in open-source llm model, it usually can't follow the instruction well, the output may be incomplete,
    so here we try to repair it and use all repair methods by default.
    typical case
        1. case sensitivity
            target: "Original Requirements"
            output: "Original requirements"
        2. special character missing
            target: [/CONTENT]
            output: [CONTENT]
        3. json format
            target: { xxx }
            output: { xxx }]
    """

    # do the repairation usually for non-openai models
    for req_key in req_keys:
        output = _repair_llm_raw_output(output=output, req_key=req_key, repair_type=repair_type)
    return output


def repair_invalid_json(output: str, error: str) -> str:
    """
    repair the situation like there are extra chars like
    error examples
        example 1. json.decoder.JSONDecodeError: Expecting ',' delimiter: line 154 column 1 (char 2765)
        example 2. xxx.JSONDecodeError: Expecting property name enclosed in double quotes: line 14 column 1 (char 266)
    """
    pattern = r"line ([0-9]+) column ([0-9]+)"

    matches = re.findall(pattern, error, re.DOTALL)
    if len(matches) > 0:
        line_no = int(matches[0][0]) - 1
        col_no = int(matches[0][1]) - 1

        # due to CustomDecoder can handle `"": ''` or `'': ""`, so convert `"""` -> `"`, `'''` -> `'`
        output = output.replace('"""', '"').replace("'''", '"')
        arr = output.split("\n")
        rline = arr[line_no]  # raw line
        line = arr[line_no].strip()
        # different general problems
        if line.endswith("],"):
            # problem, redundant char `]`
            new_line = line.replace("]", "")
        elif line.endswith("},") and not output.endswith("},"):
            # problem, redundant char `}`
            new_line = line.replace("}", "")
        elif line.endswith("},") and output.endswith("},"):
            new_line = line[:-1]
        elif (rline[col_no] in ["'", '"']) and (line.startswith('"') or line.startswith("'")) and "," not in line:
            # problem, `"""` or `'''` without `,`
            new_line = f",{line}"
        elif col_no - 1 >= 0 and rline[col_no - 1] in ['"', "'"]:
            # backslash problem like \" in the output
            char = rline[col_no - 1]
            nearest_char_idx = rline[col_no:].find(char)
            new_line = (
                rline[: col_no - 1]
                + "\\"
                + rline[col_no - 1 : col_no + nearest_char_idx]
                + "\\"
                + rline[col_no + nearest_char_idx :]
            )
        elif '",' not in line and "," not in line and '"' not in line:
            new_line = f'{line}",'
        elif not line.endswith(","):
            # problem, miss char `,` at the end.
            new_line = f"{line},"
        elif "," in line and len(line) == 1:
            new_line = f'"{line}'
        elif '",' in line:
            new_line = line[:-2] + "',"
        else:
            new_line = line

        arr[line_no] = new_line
        output = "\n".join(arr)
        logger.info(f"repair_invalid_json, raw error: {error}")

    return output


def run_after_exp_and_passon_next_retry(logger: "loguru.Logger") -> Callable[["RetryCallState"], None]:
    def run_and_passon(retry_state: RetryCallState) -> None:
        """
        RetryCallState example
            {
                "start_time":143.098322024,
                "retry_object":"<Retrying object at 0x7fabcaca25e0 (stop=<tenacity.stop.stop_after_attempt ... >)>",
                "fn":"<function retry_parse_json_text_v2 at 0x7fabcac80ee0>",
                "args":"(\"tag:[/CONTENT]\",)",  # function input args
                "kwargs":{},                     # function input kwargs
                "attempt_number":1,              # retry number
                "outcome":"<Future at xxx>",  # type(outcome.result()) = "str", type(outcome.exception()) = "class"
                "outcome_timestamp":143.098416904,
                "idle_for":0,
                "next_action":"None"
            }
        """
        if retry_state.outcome.failed:
            if retry_state.args:
                # # can't be used as args=retry_state.args
                func_param_output = retry_state.args[0]
            elif retry_state.kwargs:
                func_param_output = retry_state.kwargs.get("output", "")
            exp_str = str(retry_state.outcome.exception())

            fix_str = "try to fix it, "
            logger.warning(
                f"parse json from content inside [CONTENT][/CONTENT] failed at retry "
                f"{retry_state.attempt_number}, {fix_str}exp: {exp_str}"
            )

            repaired_output = repair_invalid_json(func_param_output, exp_str)
            retry_state.kwargs["output"] = repaired_output

    return run_and_passon


def repair_stop_after_attempt(retry_state):
    return stop_after_attempt(3)(retry_state)


@retry(
    stop=repair_stop_after_attempt,
    wait=wait_fixed(1),
    after=run_after_exp_and_passon_next_retry(logger),
)
def retry_parse_json_text(output: str) -> Union[list, dict]:
    """
    repair the json-text situation like there are extra chars like [']', '}']

    Warning
        if CONFIG.repair_llm_output is False, retry _aask_v1 {x=3} times, and the retry_parse_json_text's retry not work
        if CONFIG.repair_llm_output is True, the _aask_v1 and the retry_parse_json_text will loop for {x=3*3} times.
            it's a two-layer retry cycle
    """
    # logger.debug(f"output to json decode:\n{output}")

    # if CONFIG.repair_llm_output is True, it will try to fix output until the retry break
    parsed_data = CustomDecoder(strict=False).decode(output)

    return parsed_data


def extract_content_from_output(content: str, right_key: str = "[/CONTENT]"):
    """extract xxx from [CONTENT](xxx)[/CONTENT] using regex pattern"""

    def re_extract_content(cont: str, pattern: str) -> str:
        matches = re.findall(pattern, cont, re.DOTALL)
        for match in matches:
            if match:
                cont = match
                break
        return cont.strip()

    # TODO construct the extract pattern with the `right_key`
    raw_content = copy.deepcopy(content)
    pattern = r"\[CONTENT\]([\s\S]*)\[/CONTENT\]"
    new_content = re_extract_content(raw_content, pattern)

    # Handle potential markdown code fences ```json ... ```
    if new_content.startswith("```json"):
        new_content = new_content[len("```json\n"):-len("```")].strip()

    if not new_content.startswith("{"):
        # TODO find a more general pattern
        # # for `[CONTENT]xxx[CONTENT]xxxx[/CONTENT] situation
        logger.warning(f"extract_content try another pattern: {pattern}")
        if right_key not in new_content:
            raw_content = copy.deepcopy(new_content + "\n" + right_key)
        # # pattern = r"\[CONTENT\](\s*\{.*?\}\s*)\[/CONTENT\]"
        new_content = re_extract_content(raw_content, pattern)
        # Re-apply markdown fence removal after secondary extraction attempt
        if new_content.startswith("```json"):
            new_content = new_content[len("```json\n"):-len("```")].strip()
    else:
        if right_key in new_content:
            idx = new_content.find(right_key)
            new_content = new_content[:idx]
            new_content = new_content.strip()
            # Also remove fences if they exist here
            if new_content.startswith("```json"):
                new_content = new_content[len("```json"):].strip()


    return new_content


def extract_state_value_from_output(content: str) -> str:
    """
    For openai models, they will always return state number. But for open llm models, the instruction result maybe a
    long text contain target number, so here add a extraction to improve success rate.

    Args:
        content (str): llm's output from `Role._think`
    """
    content = content.strip()  # deal the output cases like " 0", "0\n" and so on.
    pattern = (
        r"(?<!-)[0-9]"  # TODO find the number using a more proper method not just extract from content using pattern
    )
    matches = re.findall(pattern, content, re.DOTALL)
    matches = list(set(matches))
    state = matches[0] if len(matches) > 0 else "-1"
    return state


def repair_escape_error(commands):
    """
    Repaires escape errors in command responses.
    When RoleZero parses a command, the command may contain unknown escape characters.

    This function has two steps:
    2. Transform escaped characters like '\f' to substrings like "\\\\f".

    Example:
        When the original JSON string is " {"content":"\\\\( \\\\frac{1}{2} \\\\)"} ",
        The "content" will be parsed correctly to "\( \frac{1}{2} \)".

        However, if the original JSON string is " {"content":"\( \frac{1}{2} \)"}" directly.
        It will cause a parsing error.

        To repair the wrong JSON string, the following transformations will be used:
        "\("   --->  "\\\\("
        '\f'   --->  "\\\\f"
        "\)"   --->  "\\\\)"

    """
    escape_repair_map = {
        "\a": "\\\\a",
        "\b": "\\\\b",
        "\f": "\\\\f",
        "\r": "\\\\r",
        "\t": "\\\\t",
        "\v": "\\\\v",
    }
    new_command = ""
    for index, ch in enumerate(commands):
        if ch == "\\" and index + 1 < len(commands):
            if commands[index + 1] not in ["n", '"', " "]:
                new_command += "\\"
        elif ch in escape_repair_map:
            ch = escape_repair_map[ch]
        new_command += ch
    return new_command



class BasePostProcessPlugin(object):
    model = None  # the plugin of the `model`, use to judge in `llm_postprocess`

    def run_repair_llm_output(self, output: str, schema: dict, req_key: str = "[/CONTENT]") -> Union[dict, list]:
        """
        repair steps
            1. repair the case sensitive problem using the schema's fields
            2. extract the content from the req_key pair( xx[REQ_KEY]xxx[/REQ_KEY]xx )
            3. repair the invalid json text in the content
            4. parse the json text and repair it according to the exception with retry loop
        """
        output_class_fields = list(schema["properties"].keys())  # Custom ActionOutput's fields

        content = self.run_repair_llm_raw_output(output, req_keys=output_class_fields + [req_key])
        content = self.run_extract_content_from_output(content, right_key=req_key)
        # # req_keys mocked
        content = self.run_repair_llm_raw_output(content, req_keys=[None], repair_type=RepairType.JSON)
        parsed_data = self.run_retry_parse_json_text(content)

        return parsed_data

    def run_repair_llm_raw_output(self, content: str, req_keys: list[str], repair_type: str = None) -> str:
        """inherited class can re-implement the function"""
        return repair_llm_raw_output(content, req_keys=req_keys, repair_type=repair_type)

    def run_extract_content_from_output(self, content: str, right_key: str) -> str:
        """inherited class can re-implement the function"""
        return extract_content_from_output(content, right_key=right_key)

    def run_retry_parse_json_text(self, content: str) -> Union[dict, list]:
        """inherited class can re-implement the function"""
        # logger.info(f"extracted json CONTENT from output:\n{content}")
        parsed_data = retry_parse_json_text(output=content)  # should use output=content
        return parsed_data

    def run(self, output: str, schema: dict, req_key: str = "[/CONTENT]") -> Union[dict, list]:
        """
        this is used for prompt with a json-format output requirement and outer pair key, like
            [REQ_KEY]
                {
                    "Key": "value"
                }
            [/REQ_KEY]

        Args
            outer (str): llm raw output
            schema: output json schema
            req_key: outer pair right key, usually in `[/REQ_KEY]` format
        """
        assert len(schema.get("properties")) > 0
        assert "/" in req_key

        # current, postprocess only deal the repair_llm_raw_output
        new_output = self.run_repair_llm_output(output=output, schema=schema, req_key=req_key)
        return new_output

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the entry of choosing which PostProcessPlugin to deal particular LLM model's output

from typing import Union

def llm_output_postprocess(
    output: str, schema: dict, req_key: str = "[/CONTENT]", model_name: str = None
) -> Union[dict, str]:
    """
    default use BasePostProcessPlugin if there is not matched plugin.
    """
    # TODO choose different model's plugin according to the model
    postprocess_plugin = BasePostProcessPlugin()

    result = postprocess_plugin.run(output=output, schema=schema, req_key=req_key)
    return result


TAG = "CONTENT"

class FillMode(Enum):
    CODE_FILL = "code_fill"
    XML_FILL = "xml_fill"
    SINGLE_FILL = "single_fill"

FORMAT_CONSTRAINT = f"Format: output wrapped inside [{TAG}][/{TAG}] like format example, nothing else."

SIMPLE_TEMPLATE = """
## context
{context}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow instructions of nodes, generate output and make sure it follows the format example.
"""

def dict_to_markdown(d, prefix="## ", kv_sep="\n", postfix="\n"):
    markdown_str = ""
    for key, value in d.items():
        markdown_str += f"{prefix}{key}{kv_sep}{value}{postfix}"
    return markdown_str

action_outcls_registry = dict()


class ActionNode:

    schema: Literal['raw', 'json', 'markdown'] #  default: ""
    context: str  # all the context, including all necessary info
    llm: LLM  # LLM with aask interface
    children: dict[str, "ActionNode"]

    # Action Input
    key: str
    expected_type: Type  # pydantic type
    instruction: str  # pydantic description
    example: Any  # example for In Context-Learning.

    # Action Output
    content: str
    instruct_content: BaseModel

    def __init__(
        self,
        key: str,
        expected_type: Type,
        instruction: str,
        example: Any,
        content: str = "",
        children: dict[str, "ActionNode"] = None,
        schema: str = "",
    ):
        self.key = key
        self.expected_type = expected_type
        self.instruction = instruction
        self.example = example
        self.content = content
        self.children = children if children is not None else {}
        self.schema = schema

    def __str__(self):
        return (
            f"{self.key}, {repr(self.expected_type)}, {self.instruction}, {self.example}"
            f", {self.content}, {self.children}"
        )
    def __repr__(self):
        return self.__str__()

    def add_child(self, node: "ActionNode"):
        self.children[node.key] = node

    def get_child(self, key: str) -> Union["ActionNode", None]:
        return self.children.get(key, None)

    def _get_children_mapping(self, exclude=None) -> Dict[str, Any]:
        """获得子ActionNode的字典，以key索引。"""
        exclude = exclude or []

        mapping = {}
        for key, child in self.children.items():
            if key in exclude:
                continue
            mapping[key] = (child.expected_type, Field(default=child.example, description=child.instruction))
        return mapping

    def _get_self_mapping(self) -> Dict[str, Tuple[Type, Any]]:
        """get self key: type mapping"""
        return {self.key: (self.expected_type, ...)}

    def get_mapping(self, mode="children", exclude=None) -> Dict[str, Tuple[Type, Any]]:
        """get key: type mapping under mode"""
        if mode == "children" or (mode == "auto" and self.children):
            return self._get_children_mapping(exclude=exclude)
        return {} if exclude and self.key in exclude else self._get_self_mapping()

    @classmethod
    def create_model_class(cls, class_name: str, mapping: Dict[str, Tuple[Type, Any]]):
        """基于pydantic v2的模型动态生成，用来检验结果类型正确性"""

        def check_fields(cls, values):
            all_fields = set(mapping.keys())
            required_fields = set()
            for k, v in mapping.items():
                type_v, field_info = v
                if ActionNode.is_optional_type(type_v):
                    continue
                required_fields.add(k)

            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")

            unrecognized_fields = set(values.keys()) - all_fields
            if unrecognized_fields:
                logger.warning(f"Unrecognized fields: {unrecognized_fields}")
            return values

        validators = {"check_missing_fields_validator": model_validator(mode="before")(check_fields)}

        new_fields = {}
        for field_name, field_value in mapping.items():
            if isinstance(field_value, dict):
                # 对于嵌套结构，递归创建模型类
                nested_class_name = f"{class_name}_{field_name}"
                nested_class = cls.create_model_class(nested_class_name, field_value)
                new_fields[field_name] = (nested_class, ...)
            else:
                new_fields[field_name] = field_value

        new_class = create_model(class_name, __validators__=validators, **new_fields)
        return new_class

    def create_class(self, mode: str = "auto", class_name: str = None, exclude=None):
        class_name = class_name if class_name else f"{self.key}_AN"
        mapping = self.get_mapping(mode=mode, exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def _create_children_class(self, exclude=None):
        """使用object内有的字段直接生成model_class"""
        class_name = f"{self.key}_AN"
        mapping = self._get_children_mapping(exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:
        """将当前节点与子节点都按照node: format的格式组织成字典"""
        nodes = self._to_dict(format_func=format_func, mode=mode, exclude=exclude)
        if not isinstance(nodes, dict):
            nodes = {self.key: nodes}
        return nodes

    def _to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:
        """将当前节点与子节点都按照node: format的格式组织成字典"""

        # 如果没有提供格式化函数，则使用默认的格式化函数
        format_func = format_func or (lambda node: node.instruction)

        # 使用提供的格式化函数来格式化当前节点的值
        formatted_value = format_func(self)

        # 创建当前节点的键值对
        if (mode == "children" or mode == "auto") and self.children:
            node_value = {}
        else:
            node_value = formatted_value

        if mode == "root":
            return {self.key: node_value}

        exclude = exclude or []
        for child_key, child_node in self.children.items():
            if child_key in exclude:
                continue
            child_dict = child_node._to_dict(format_func, mode, exclude)
            node_value[child_key] = child_dict
        return node_value

    def keys(self, mode: str = "auto") -> list:
        if mode == "children" or (mode == "auto" and self.children):
            keys = []
        else:
            keys = [self.key]
        if mode == "root":
            return keys

        for _, child_node in self.children.items():
            keys.append(child_node.key)
        return keys

    def compile_to(self, i: Dict, schema, kv_sep) -> str:
        if schema == "json":
            return json.dumps(i, indent=4, ensure_ascii=False)
        elif schema == "markdown":
            return dict_to_markdown(i, kv_sep=kv_sep)
        else:
            return str(i)

    def tagging(self, text, schema, tag="") -> str:
        if not tag:
            return text
        return f"[{tag}]\n{text}\n[/{tag}]"

    def _compile_f(self, schema, mode, tag, format_func, kv_sep, exclude=None) -> str:
        nodes = self.to_dict(format_func=format_func, mode=mode, exclude=exclude)
        text = self.compile_to(nodes, schema, kv_sep)
        return self.tagging(text, schema, tag)

    def compile_instruction(self, schema="markdown", mode="children", tag="", exclude=None) -> str:
        """compile to raw/json/markdown template with all/root/children nodes"""
        format_func = lambda i: f"{i.expected_type}  # {i.instruction}"
        return self._compile_f(schema, mode, tag, format_func, kv_sep=": ", exclude=exclude)

    def compile_example(self, schema="json", mode="children", tag="", exclude=None) -> str:
        """compile to raw/json/markdown examples with all/root/children nodes"""

        # 这里不能使用f-string，因为转译为str后再json.dumps会额外加上引号，无法作为有效的example
        # 错误示例："File list": "['main.py', 'const.py', 'game.py']", 注意这里值不是list，而是str
        format_func = lambda i: i.example
        return self._compile_f(schema, mode, tag, format_func, kv_sep="\n", exclude=exclude)

    def compile(self, context, schema="json", mode="children", template=SIMPLE_TEMPLATE, exclude=[]) -> str:
        """
        mode: all/root/children
            mode="children": 编译所有子节点为一个统一模板，包括instruction与example
            mode="all": NotImplemented
            mode="root": NotImplemented
        schmea: raw/json/markdown
            schema="raw": 不编译，context, lang_constaint, instruction
            schema="json"：编译context, example(json), instruction(markdown), constraint, action
            schema="markdown": 编译context, example(markdown), instruction(markdown), constraint, action
        """
        if schema == "raw":
            return f"{context}\n\n## Actions\n{self.instruction}"

        # FIXME: json instruction会带来格式问题，如："Project name": "web_2048  # 项目名称使用下划线",
        # compile example暂时不支持markdown
        instruction = self.compile_instruction(schema="markdown", mode=mode, exclude=exclude)
        example = self.compile_example(schema=schema, tag=TAG, mode=mode, exclude=exclude)
        # nodes = ", ".join(self.to_dict(mode=mode).keys())
        constraint = FORMAT_CONSTRAINT

        prompt = template.format(
            context=context,
            example=example,
            instruction=instruction,
            constraint=constraint,
        )
        return prompt

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
    )
    async def _aask_v1(
        self,
        prompt: str,
        output_class_name: str,
        output_data_mapping: dict,
        images: Optional[Union[str, list[str]]] = None,
        system_msgs: Optional[list[str]] = None,
        schema="markdown",  # compatible to original format
        timeout=10,
    ) -> (str, BaseModel):
        """Use ActionOutput to wrap the output of aask"""
        if images and len(images) > 10:
            logger.warning(f"_aask_v1: Number of images exceeds 10: {len(images)}. message: {prompt}")
        if len(prompt) > 2000:
            logger.error(f"_aask_v1: Prompt exceeds 2000 characters: {len(prompt)}. message: {prompt}. len(images): {len(images)}")
            raise
        content = await self.llm.aask(prompt, system_msgs, images=images, timeout=timeout)
        logger.debug(f"llm raw output:\n{content}")
        output_class = self.create_model_class(output_class_name, output_data_mapping)

        assert schema == "json"
        parsed_data = llm_output_postprocess(
            output=content, schema=output_class.model_json_schema(), req_key=f"[/{TAG}]"
        )

        logger.debug(f"parsed_data:\n{parsed_data}")
        instruct_content = output_class(**parsed_data)
        return content, instruct_content

    def get(self, key):
        return self.instruct_content.model_dump()[key]

    def set_recursive(self, name, value):
        setattr(self, name, value)
        for _, i in self.children.items():
            i.set_recursive(name, value)

    def set_llm(self, llm):
        self.set_recursive("llm", llm)

    def set_context(self, context):
        self.set_recursive("context", context)

    async def simple_fill(
        self, schema, mode, images: Optional[Union[str, list[str]]] = None, timeout=10, exclude=None
    ):
        prompt = self.compile(context=self.context, schema=schema, mode=mode, exclude=exclude)
        if schema != "raw":
            mapping = self.get_mapping(mode, exclude=exclude)
            class_name = f"{self.key}_AN"
            content, scontent = await self._aask_v1(
                prompt, class_name, mapping, images=images, schema=schema, timeout=timeout
            )
            self.content = content
            self.instruct_content = scontent
        else:
            self.content = await self.llm.aask(prompt)
            self.instruct_content = None

        return self

    def get_field_name(self):
        """
        Get the field name from the Pydantic model associated with this ActionNode.
        """
        model_class = self.create_class()
        fields = model_class.model_fields

        # Assuming there's only one field in the model
        if len(fields) == 1:
            return next(iter(fields))

        # If there are multiple fields, we might want to use self.key to find the right one
        return self.key

    def get_field_names(self):
        """
        Get the field names associated with this ActionNode's Pydantic model.
        """
        model_class = self.create_class()
        return model_class.model_fields.keys()

    def get_field_types(self):
        """
        Get the field types associated with this ActionNode's Pydantic model.
        """
        model_class = self.create_class()
        return {field_name: field.annotation for field_name, field in model_class.model_fields.items()}

    def xml_compile(self, context):
        """
        Compile the prompt to make it easier for the model to understand the xml format.
        """
        field_names = self.get_field_names()
        # Construct the example using the field names
        examples = []
        for field_name in field_names:
            examples.append(f"<{field_name}>content</{field_name}>")

        # Join all examples into a single string
        example_str = "\n".join(examples)
        # Add the example to the context
        context += f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""
        return context

    async def single_fill(self, context: str, images: Optional[Union[str, list[str]]] = None) -> Dict[str, str]:
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, images=images)
        result = {field_name: content}
        return result

    async def xml_fill(self, context: str, images: Optional[Union[str, list[str]]] = None) -> Dict[str, Any]:
        """
        Fill context with XML tags and convert according to field types, including string, integer, boolean, list and dict types
        """
        field_names = self.get_field_names()
        field_types = self.get_field_types()

        extracted_data: Dict[str, Any] = {}
        content = await self.llm.aask(context, images=images)

        for field_name in field_names:
            pattern = rf"<{field_name}>(.*?)</{field_name}>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                raw_value = match.group(1).strip()
                field_type = field_types.get(field_name)

                if field_type == str:
                    extracted_data[field_name] = raw_value
                elif field_type == int:
                    try:
                        extracted_data[field_name] = int(raw_value)
                    except ValueError:
                        extracted_data[field_name] = 0  # 或者其他默认值
                elif field_type == bool:
                    extracted_data[field_name] = raw_value.lower() in ("true", "yes", "1", "on", "True")
                elif field_type == list:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], list):
                            raise ValueError
                    except:
                        extracted_data[field_name] = []  # 默认空列表
                elif field_type == dict:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], dict):
                            raise ValueError
                    except:
                        extracted_data[field_name] = {}  # 默认空字典
                else:
                    extracted_data[field_name] = raw_value
            else:
                print(context)
                print(field_names)
                print(field_types)

        return extracted_data

    async def fill(
        self,
        *,
        context,
        llm,
        schema="json",
        mode="auto",
        strgy="simple",
        images: list[Image.Image] = list(),
        timeout=10,
        exclude=[],
        function_name: str = None,
    ):
        """Fill the node(s) with mode.

        :param context: 
        :param llm: Large Language Model with pre-defined system message.
        :param schema: json/markdown, determine example and output format.
         - raw: free form text
         - json: it's easy to open source LLM with json format
         - markdown: when generating code, markdown is always better
        :param mode: auto/children/root
         - auto: automated fill children's nodes and gather outputs, if no children, fill itself
         - children: fill children's nodes and gather outputs
         - root: fill root's node and gather output
        :param strgy: simple/complex
         - simple: run only once
         - complex: run each node
        :param images: the list of image url or base64 for gpt4-v
        :param timeout: Timeout for llm invocation.
        :param exclude: The keys of ActionNode to exclude.
        :return: self
        """
        
        if isinstance(context, tuple):
            for x in context:
                assert isinstance(x, str) or isinstance(x, Image.Image)
            assert not images
            images = [x for x in context if isinstance(x, Image.Image)]
            context = ' '.join(x if isinstance(x, str) else '<image>' for x in context)

        self.set_llm(llm)
        self.set_context(context)
        if self.schema:
            schema = self.schema

        elif mode == FillMode.XML_FILL.value:
            context = self.xml_compile(context=self.context)
            result = await self.xml_fill(context, images=images)
            self.instruct_content = self.dantic(**result)
            return self

        elif mode == FillMode.SINGLE_FILL.value:
            result = await self.single_fill(context, images=images)
            self.instruct_content = self.create_class()(**result)
            return self

        if strgy == "simple":
            return await self.simple_fill(schema=schema, mode=mode, images=images, timeout=timeout, exclude=exclude)
        elif strgy == "complex":
            tmp = {}
            for _, i in self.children.items():
                if exclude and i.key in exclude:
                    continue
                child = await i.simple_fill(schema=schema, mode=mode, images=images, timeout=timeout, exclude=exclude)
                tmp.update(child.instruct_content.model_dump())
            cls = self._create_children_class()
            self.instruct_content = cls(**tmp)
            return self

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], key: str = None):
        """
        Creates an ActionNode tree from a Pydantic model.

        Args:
            model (Type[BaseModel]): The Pydantic model to convert.

        Returns:
            ActionNode: The root node of the created ActionNode tree.
        """
        key = key or model.__name__
        root_node = cls(key=key, expected_type=Type[model], instruction="", example="")

        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            description = field_info.description
            default = field_info.default

            # Recursively handle nested models if needed
            if not isinstance(field_type, typing._GenericAlias) and issubclass(field_type, BaseModel):
                child_node = cls.from_pydantic(field_type, key=field_name)
            else:
                child_node = cls(key=field_name, expected_type=field_type, instruction=description, example=default)

            root_node.add_child(child_node)

        root_node.dantic = model
        return root_node

    @staticmethod
    def is_optional_type(tp) -> bool:
        """Return True if `tp` is `typing.Optional[...]`"""
        if typing.get_origin(tp) is Union:
            args = typing.get_args(tp)
            non_none_types = [arg for arg in args if arg is not type(None)]
            return len(non_none_types) == 1 and len(args) == 2
        return False


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")
def parse_bbox_string(v: str) -> Tuple[int, int, int, int]:
    try:
        numbers = v.split()
        if len(numbers) != 4:
            raise ValueError("Expected four numbers in bbox string")
        return tuple(int(num) for num in numbers)
    except (ValueError, TypeError):
        raise ValueError("Invalid bbox format; expected 'num num num num'")
BBoxType = Annotated[Tuple[int, int, int, int], BeforeValidator(parse_bbox_string)]

class CropOp(BaseModel):
    thought: str = Field(default="", description="Thoughts on what crop may be sufficient.")
    bbox: BBoxType = Field(..., description="a crop containing all relevant information, in x y x y format, idx from 0 to 1000")


def custom(input, model='gemini-2.0-flash', dna=GenerateOp):
    return asyncio.run(ActionNode.from_pydantic(dna).fill(context=input, llm=LLM(model=model)))

def test_llm_output_postprocess():
    print(llm_output_postprocess(output='''[CONTENT]
```json
{
    "response": ""
}
```
[/CONTENT]''', schema={'properties': {'request': str}}))
    assert llm_output_postprocess(output='''[CONTENT]
```json
{
    "response": ""
}
```
[/CONTENT]''', schema={'properties': {'request': str}}) == {'response': ''}
def test_custom():
    print(custom('hi! '))



class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, context, mode=None, **extra_kwargs):
        fill_kwargs = {"context": context, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)

        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

def crop1000(image, box: tuple):
    x, y = image.width, image.height
    return image.crop((box[1] / 1000 * x, box[0] / 1000 * y, box[3] / 1000 * x, box[2] / 1000 * y))
    

class Custom(Operator):
    def __init__(self, llm: LLM = None, name: str = "Custom"):
        super().__init__(llm or LLM(), name)

    async def __call__(self, input, pydantic_model=GenerateOp, mode: Literal["single_fill", "xml_fill", "code_fill"] = "single_fill"):
        return await self._fill_node(pydantic_model, input, mode=mode)

CROP_PROMPT = """We are trying to remove irrelevant information from an image.
Given a question and its image, please output the bounding box within which the information necessary for answering this question entirely lies.

question: {question}
"""
class Crop(Operator):
    def __init__(self, llm: LLM = None, name: str = "Crop"):
        super().__init__(llm or LLM(), name)
    
    async def __call__(self, image, question):
        response = await self._fill_node(CropOp, CROP_PROMPT.format(question=question), images=[image],mode='xml_fill')
        bbox = response['bbox']
        ret = crop1000(image, bbox)
        uid = uuid.uuid4()
        ret.save(f"crop_{uid}.png")
        return ret

def test_crop():
    crop = Crop()()
import requests
import io
import numpy as np
from PIL import Image
from som import inference_sam_m2m_auto
import functools
from gradio_client import Client
import requests
import os

def call_mask_api(image):
    server_url = "http://localhost:7861"
    client = Client(server_url)
    filename = client.predict(image=image, api_name="/predict")

    download_url = f"{server_url}/file={filename}"

    response = requests.get(download_url)
    assert response.status_code == 200, f"Failed to download the file. Status code: {response.status_code}"
    
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"File downloaded successfully: {filename}")
    
    data = np.load(filename)
    segmentations = data['segmentations']
    areas = data['areas']
    bboxes = data['bboxes']
    predicted_ious = data['predicted_ious']
    
    num_masks = segmentations.shape[0]
    masks = []
    for i in range(num_masks):
        mask = {
            'segmentation': segmentations[i],
            'area': areas[i],
            'bbox': bboxes[i],
            'predicted_iou': predicted_ious[i]
        }
        masks.append(mask)
    print(f"Reconstructed {len(masks)} masks")
    return masks


def test_api():
    image = Image.open('/mnt/home/jkp/hack/diane/data/zerobench_images/zerobench/example_21_image_0.png')
    ret = call_mask_api(image)
    print(type(ret))
    print(ret.keys())
    for x, y in ret.items():
        print(x, y)

if __name__ == '__main__':
    test_api()

def image_mask_to_alpha(image, mask):
    masked = image.copy()
    alpha_mask = Image.fromarray(mask['segmentation'] * 255, mode='L')
    masked.putalpha(alpha_mask)
    return masked

def alpha_to_mask(image):
    segmentation = image.split()[-1]
    bbox = segmentation.getbbox()
    area = area(image)
    return {
        'segmentation': segmentation,
        'bbox': bbox,
        'area': area,
    }

def sam_operator(image):
    ret = call_mask_api(image)
    sorted_anns = sorted(ret, key=(lambda x: x['area']), reverse=True)
    sorted_anns = sorted_anns[:10]
    # a mask of all area not being masked:
    mask_neg = image.copy().to('RGBA')
    for ann in sorted_anns:
        mask = ann['segmentation']
        mask_neg.putalpha(Image.fromarray((1 - mask) * 255, mode='L'))
    sorted_anns = map(lambda ann: image_mask_to_alpha(image, ann), sorted_anns)
    sorted_anns.append(mask_neg)
    return sorted_anns

def combine(images: list[Image.Image]):
    return functools.reduce(lambda a, b: a.alpha_composite(b), images, initial=Image.new('RGBA', images[0].size, (0, 0, 0, 0)))

def sam2_alpha(image):
    ret = call_mask_api(image)
    print(type(ret))
    return list(map(alpha_to_mask, image, ret))

def set_of_mask(image):
    if isinstance(image, list):
        assert isinstance(image[0], Image.Image)
        ret = map(alpha_to_mask, image)
        image = combine(image)
    elif isinstance(image, Image.Image):
        ret = call_mask_api(image)
    return inference_sam_m2m_auto(image=image, outputs=ret)


operators = {
    'Custom': Custom(),
    'Crop': Crop(),
    'SAM': sam_operator,
    'SoM': set_of_mask,
}

operators_doc = {
    'Custom': {
        'description': "A simple vlm call.",
        'interface': "custom(input: str | tuple[str | Image.Image]):"
    },
}
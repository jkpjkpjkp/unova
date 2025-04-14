import re

def xml_compile(field_names: list[str]):
    examples = []
    for field_name in field_names:
        examples.append(f"<{field_name}>content</{field_name}>")
    example_str = "\n".join(examples)
    return f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""

def xml_extract(content: str, field_names: list[str], field_types: dict[str, type]) -> dict[str, any]:
    extracted_data: dict[str, any] = {}

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
                    extracted_data[field_name] = 0
            elif field_type == bool:
                extracted_data[field_name] = raw_value.lower() in ("true", "yes", "1", "on", "True")
            elif field_type == list:
                try:
                    extracted_data[field_name] = eval(raw_value)
                    if not isinstance(extracted_data[field_name], list):
                        raise ValueError
                except:
                    extracted_data[field_name] = []
            elif field_type == dict:
                try:
                    extracted_data[field_name] = eval(raw_value)
                    if not isinstance(extracted_data[field_name], dict):
                        raise ValueError
                except:
                    extracted_data[field_name] = {}

    return extracted_data
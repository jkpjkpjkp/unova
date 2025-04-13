class ANode:

    def xml_compile(self, field_names: list[str]):
        examples = []
        for field_name in field_names:
            examples.append(f"<{field_name}>content</{field_name}>")
        example_str = "\n".join(examples)
        return f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""
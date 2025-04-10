
###### UNDONE BELOW

import re
import json
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import asyncio

# Helper function to extract JSON from LLM output
def extract_json(content: str) -> Optional[dict]:
    """Extract the first valid JSON object from a string."""
    # Try finding content between the first { and last }
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and start < end:
        json_str = content[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    # Fallback to regex for a JSON-like structure
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

# Helper function to parse markdown key-value pairs
def parse_markdown(content: str, fields: list[str]) -> dict:
    """Parse markdown content into a dictionary based on expected fields."""
    parsed_data = {}
    for field in fields:
        pattern = rf"- {field}:\s*(.*)"
        match = re.search(pattern, content)
        if match:
            parsed_data[field] = match.group(1).strip()
    return parsed_data

async def simple_fill(
    context: str,
    output_class: Type[BaseModel],
    schema: str = "json",
    images: Optional[Union[str, list[str]]] = None,
    timeout: int = 180,
    model: str = "gpt-4"
) -> tuple[str, Optional[BaseModel]]:
    """
    Generate a prompt based on context and schema, call OpenAI directly, and parse the output into a Pydantic model.

    Args:
        context: The input context or instruction for the LLM.
        output_class: Pydantic model class defining the expected output structure.
        schema: Output format ("json", "markdown", or "raw"). Default is "json".
        images: Optional image URLs for multimodal input.
        timeout: Request timeout in seconds. Default is 180.
        model: OpenAI model to use (e.g., "gpt-4"). Default is "gpt-4".

    Returns:
        Tuple of (raw LLM content, parsed Pydantic model instance or None).
    """
    client = OpenAI()  # Assumes API key is set via environment variable OPENAI_API_KEY

    # Construct prompt based on schema
    if schema == "json":
        schema_str = json.dumps(output_class.model_json_schema(), indent=2)
        prompt = f"{context}\n\nProvide the response in this JSON format:\n{schema_str}"
    elif schema == "markdown":
        fields = "\n".join(f"- {name}: {field.annotation}" for name, field in output_class.model_fields.items())
        prompt = f"{context}\n\nProvide the response with these fields:\n{fields}"
    else:
        prompt = context

    # Prepare message content (supporting text and images)
    if images:
        if isinstance(images, str):
            images = [images]
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})
    else:
        content = prompt

    messages = [{"role": "user", "content": content}]

    # Call OpenAI API asynchronously
    def sync_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            request_timeout=timeout
        )
    response = await asyncio.to_thread(sync_call)
    content = response.choices[0].message.content

    # Parse output based on schema
    if schema == "json":
        parsed_data = extract_json(content)
        if parsed_data:
            try:
                return content, output_class(**parsed_data)
            except ValidationError:
                return content, None
        return content, None
    elif schema == "markdown":
        fields = list(output_class.model_fields.keys())
        parsed_data = parse_markdown(content, fields)
        if parsed_data:
            try:
                return content, output_class(**parsed_data)
            except ValidationError:
                return content, None
        return content, None
    return content, None

async def xml_fill(
    context: str,
    output_class: Type[BaseModel],
    images: Optional[Union[str, list[str]]] = None,
    timeout: int = 180,
    model: str = "gpt-4"
) -> tuple[str, BaseModel]:
    """
    Generate a prompt with XML tags based on the Pydantic model, call OpenAI directly, and extract structured data.

    Args:
        context: The input context or instruction for the LLM.
        output_class: Pydantic model class defining the expected output structure.
        images: Optional image URLs for multimodal input.
        timeout: Request timeout in seconds. Default is 180.
        model: OpenAI model to use (e.g., "gpt-4"). Default is "gpt-4".

    Returns:
        Tuple of (raw LLM content, parsed Pydantic model instance).
    """
    client = OpenAI()  # Assumes API key is set via environment variable OPENAI_API_KEY

    # Generate XML-tagged prompt
    field_names = list(output_class.model_fields.keys())
    examples = "\n".join(f"<{name}>content</{name}>" for name in field_names)
    prompt = f"{context}\n\n### Response format (strict): Use XML tags with matching <tag></tag>:\n{examples}"

    # Prepare message content (supporting text and images)
    if images:
        if isinstance(images, str):
            images = [images]
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})
    else:
        content = prompt

    messages = [{"role": "user", "content": content}]

    # Call OpenAI API asynchronously
    def sync_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            request_timeout=timeout
        )
    response = await asyncio.to_thread(sync_call)
    content = response.choices[0].message.content

    # Extract data from XML tags
    extracted_data: Dict[str, Any] = {}
    for name, field in output_class.model_fields.items():
        pattern = rf"<{name}>(.*?)</{name}>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if field.annotation == str:
                extracted_data[name] = value
            elif field.annotation == int:
                extracted_data[name] = int(value) if value.isdigit() else 0
            # Add more type conversions as needed

    return content, output_class(**extracted_data)
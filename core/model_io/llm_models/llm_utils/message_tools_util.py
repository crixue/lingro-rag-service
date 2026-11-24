from typing import Any


def to_flat_tool_message(tool: Any) -> dict:
    """
    Convert an OpenAI-style tool:
      {"type":"function","function":{"name":..., "description":..., "parameters":...}}
    into a flat tool:
      {"type":"function","name":..., "description":..., "parameters":...}
    """
    if isinstance(tool, dict):
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            return {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        return tool

    # If tool is a LangChain tool object, best-effort extraction
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", None)
    args_schema = getattr(tool, "args_schema", None)

    parameters = None
    if args_schema is not None:
        try:
            parameters = args_schema.model_json_schema()
        except Exception:
            parameters = None

    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
    }
import json
from typing import Optional

from pydantic import BaseModel, Field

from utils.common_util import remove_none_value_from_dict


class OpenAIStandardFunctionBO(BaseModel):
    """OpenAI standard function business object.
    e.g.:
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
    """

    class ParametersDefinition(BaseModel):
        """Parameters definition."""
        type: str = Field(default="object", description="The type of the parameters.", required=False)
        properties: dict = Field(default={}, description="The properties of the parameters.", required=False)
        required: list = Field(default=[], description="The required properties of the parameters.", required=False)

    name: str = Field(default=None, description="The name of the function.", required=True)
    description: str = Field( default=None, description="The description of the function.", required=True)
    parameters: Optional[ParametersDefinition] = None



if __name__ == '__main__':
    json_str = {
    "id":"f77b31",
    "createdAt":"2024-01-01 22:29:11",
    "updatedAt":"2024-01-01 22:29:11",
    "lang":"cn",
    "ver":"v1",
    "status":1,
    "nameAlias":"当前时间",
    "descriptionAlias":"获取当前北京时间",
    "avatar":"",
    "name":"get_current_datetime_f77b31",
    "description":"Get the current Beijing time.",
    "supportedModels":[
        "qwen",
        "chatglm"
    ]
}
    var = OpenAIStandardFunctionBO(**json_str)
    print(json.dumps(remove_none_value_from_dict(var.dict()), indent=4, skipkeys=True))

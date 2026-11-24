from fastapi import APIRouter
from tools_definitions import ToolsDefinition
from utils.response_util import server_response

router = APIRouter()

@router.get('/plugins/v1/supported_functions')
def plugins_supported_functions(lang: str = 'cn'):
    tools_definition = ToolsDefinition(lang=lang)
    supported_functions = tools_definition.list_all_supported_function_definitions()
    return server_response(data=supported_functions)
from fastapi import APIRouter, Depends
from typing import Any, Optional

from tools_definitions.mcp.mcp_tools import list_server_tools_async, get_prompt, list_prompts_async, \
    list_resources_async, list_resource_templates_async
from controllers.vo.mcp_connection import MCPServer
from utils.response_util import server_response

router = APIRouter()


@router.post('/mcp-servers/v1/tools/list')
async def list_tools(server_config: MCPServer):
    """
    获取指定服务器的工具列表
    :param server_config: MCP服务器配置
    :param server_name: 服务器名称（可选）
    :return: 工具列表
    """

    data = await list_server_tools_async(server_config)
    return server_response(data=data)

@router.post('/mcp-servers/v1/prompts/list')
async def list_prompts(server_config: MCPServer):
    """
    获取指定服务器的工具列表
    :param server_config: MCP服务器配置
    :param server_name: 服务器名称（可选）
    :return: 工具列表
    """

    data = await list_prompts_async(server_config)
    return server_response(data=data)

@router.post('/mcp-servers/v1/prompt/get')
async def fetch_prompt(server_config: MCPServer):
    """
    获取指定服务器的提示信息
    :param server_config: MCP服务器配置
    :param server_name: 服务器名称
    :param prompt_name: 提示名称
    :param arguments: 提示参数（可选）
    :return: 提示内容
    """
    data = await get_prompt(server_config)
    return server_response(data=data)


@router.post('/mcp-servers/v1/resources/list')
async def list_mcp_servers_resources(server_config: MCPServer):
    """
    获取指定服务器的资源信息
    :param server_config: MCP服务器配置
    :param server_name: 服务器名称（可选）
    :return: 资源信息列表
    """
    data = await list_resources_async(server_config)
    return server_response(data=data)


@router.post('/mcp-servers/v1/resources/templates/list')
async def list_resources_template(server_config: MCPServer):
    data = await list_resource_templates_async(server_config)
    return server_response(data=data)
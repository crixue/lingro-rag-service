import asyncio
from typing import Any, Optional

from exceptiongroup import ExceptionGroup
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, StreamableHttpConnection, \
    WebsocketConnection
from langchain_mcp_adapters.tools import _list_all_tools
from mcp import McpError

from configs.model_config import logger, MCP_REQUEST_CONFIG
from controllers.vo.mcp_connection import MCPServer
from service_communication.access_token_manager import AccessTokenManager

HAS_ERROR = "hasError"
ERR_INFO = "errInfo"


async def convert_connections(mcp_connections: dict, is_hosted: Optional[bool]) -> dict[str, StdioConnection| SSEConnection | StreamableHttpConnection | WebsocketConnection]:
    converted_connections = {}
    access_token_manager = AccessTokenManager()
    for name, mcp_connection in mcp_connections.items():
        connection = convert_connection(mcp_connection.model_dump())
        if is_hosted:
            # 如果是托管的MCP服务器，添加access_token
            if 'headers' in connection:
                access_token = await access_token_manager.get_access_token_async()
                connection["headers"] = {
                    "Authorization": f"Bearer {access_token}"
                }
        converted_connections[name] = connection
        logger.debug(f"[convert_connections]-Converted connection for {name}: {connection}")
    return converted_connections

def convert_connection(mcp_connection: dict):
    """
    将mcp_connection.py中的连接类型转换为sessions.py中的连接类型。

    :param mcp_connection: 来自mcp_connection.py的连接配置
    :return: 转换后的sessions.py连接配置
    """
    transport = mcp_connection.get("transport")

    if transport == "stdio":
        return StdioConnection(
            **mcp_connection,
        )
    elif transport == "sse":
        return SSEConnection(
            **mcp_connection,
        )
    elif transport == "streamable_http":
        return StreamableHttpConnection(
            **mcp_connection,

        )
    elif transport == "websocket":
        return WebsocketConnection(
            **mcp_connection,
        )
    else:
        raise ValueError(f"Unsupported transport type: {transport}")


async def handle_task_with_timeout(server_name: str, task, timeout: float) -> tuple[str, dict[str, Any]]:
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        return server_name, {
            HAS_ERROR: True,
            ERR_INFO: "The request timed out. Please try again later."
        }

async def list_server_tools_async(server_config: MCPServer) -> dict[str, dict[str, Any]]:
    connections = await convert_connections(server_config.mcp_servers, server_config.hosted)
    client = MultiServerMCPClient(connections)
    tools_mapping = {}
    server_name_list = server_config.query_server_names
    if server_name_list is None or len(server_name_list) == 0:
        server_name_list = server_config.mcp_servers.keys()
    # for name in server_name_list:
    #     try:
    #         async with client.session(name) as session:
    #             tools = await _list_all_tools(session)
    #             tools_mapping[name] = {
    #                 HAS_ERROR: False,
    #                 "tools": tools,
    #             }
    #     except ExceptionGroup as eg:
    #         logger.error(f"[list_server_tools_async]-Error fetching tools for server {name}: {eg}")
    #         err_msg = f"Error fetching tools for server {name}: \n\n"
    #         for e in eg.exceptions:
    #             err_msg += f"{str(e)};\\n"
    #         tools_mapping[name] = {
    #             HAS_ERROR: True,
    #             ERR_INFO: err_msg
    #         }
    #     except Exception as e:
    #         logger.error(f"[list_server_tools_async]-Error fetching tools for server {name}: {e}")
    #         tools_mapping[name] = {
    #             HAS_ERROR: True,
    #             ERR_INFO: str(e)
    #         }

    async def fetch_tools_for_server(name: str) -> tuple[str, dict[str, Any]]:
        try:
            async with client.session(name) as session:
                tools = await _list_all_tools(session)
                return name, {
                    HAS_ERROR: False,
                    "tools": tools,
                }
        except ExceptionGroup as eg:
            logger.error(f"[list_server_tools_async]-Error fetching tools for server {name}: {eg}")
            err_msg = f"Error fetching tools for server {name}: \n\n"
            for e in eg.exceptions:
                err_msg += f"{str(e)};\\n"
            return name, {
                HAS_ERROR: True,
                ERR_INFO: err_msg
            }
        except Exception as e:
            logger.error(f"[list_server_tools_async]-Error fetching tools for server {name}: {e}")
            return name, {
                HAS_ERROR: True,
                ERR_INFO: str(e)
            }

    tasks = [handle_task_with_timeout(name, fetch_tools_for_server(name),
                                      MCP_REQUEST_CONFIG['request_timeout']) for name in server_name_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"[list_server_tools_async]-Unexpected error: {result}")
        else:
            name, tools_data = result
            tools_mapping[name] = tools_data

    return tools_mapping


async def list_prompts_async(server_config: MCPServer) -> dict[str, dict[str, Any]]:
    connections = await convert_connections(server_config.mcp_servers, server_config.hosted)
    client = MultiServerMCPClient(connections)
    templates_mapping = {}
    server_name_list = server_config.query_server_names
    if server_name_list is None or len(server_name_list) == 0:
        server_name_list = server_config.mcp_servers.keys()
    for name in server_name_list:
        try:
            async with client.session(name) as session:
                current_cursor: str | None = None
                all_prompts = []
                while True:
                    try:
                        prompts_page = await session.list_prompts(cursor=current_cursor)
                    except McpError as e:
                        logger.error(f"[list_prompts_async]-Error fetching prompts for server {name}: {e}")
                        break
                    if prompts_page.prompts:
                        all_prompts.extend(prompts_page.prompts)
                    if not prompts_page.nextCursor:
                        break
                    current_cursor = prompts_page.nextCursor
                templates_mapping[name] = {
                    HAS_ERROR: False,
                    "prompts": all_prompts,
                }
        except ExceptionGroup as eg:
            logger.error(f"[list_prompts_async]-Error fetching prompts for server {name}: {eg}")
            err_msg = f"Error fetching tools for server {name}: \n\n"
            for e in eg.exceptions:
                err_msg += f"{str(e)};\\n"
            templates_mapping[name] = {
                HAS_ERROR: True,
                ERR_INFO: err_msg
            }
        except Exception as e:
            logger.error(f"[list_prompts_async]-Error fetching prompts for server {name}: {e}")
            templates_mapping[name] = {
                HAS_ERROR: True,
                ERR_INFO: str(e)
            }
    return templates_mapping


async def get_prompt(server_config: MCPServer):
    templates_mapping = {}
    name = server_config.query_server_name or server_config.mcp_servers.keys()[0]
    try:
        client = MultiServerMCPClient(server_config.mcp_servers)
        async with client.session(server_config.query_server_name) as session:
            response = await session.get_prompt(server_config.prompt_name, server_config.arguments)
            return response
    except ExceptionGroup as eg:
        logger.error(f"[get_prompt]-Error prompt for server {eg}")
        err_msg = f"Error fetching prompt: \n\n"
        for e in eg.exceptions:
            err_msg += f"{str(e)};\\n"
        templates_mapping[name] = {
            HAS_ERROR: True,
            ERR_INFO: err_msg
        }
    except Exception as e:
        logger.error(f"[get_prompt]-Error fetching prompt: {e}")
        return {
            HAS_ERROR: True,
            ERR_INFO: str(e)
        }


async def list_resources_async(server_config: MCPServer) -> dict[str, dict[str, Any]]:
    connections = await convert_connections(server_config.mcp_servers, server_config.hosted)
    client = MultiServerMCPClient(connections)
    resources_mapping = {}
    server_name_list = server_config.query_server_names
    if server_name_list is None or len(server_name_list) == 0:
        server_name_list = server_config.mcp_servers.keys()
    for name in server_name_list:
        try:
            async with client.session(name) as session:
                current_cursor: str | None = None
                all_resources = []
                while True:
                    try:
                        resources_page = await session.list_resources(cursor=current_cursor)
                    except McpError as e:
                        logger.error(f"[list_resources_async]-Error fetching resources for server {name}: {e}")
                        break
                    if resources_page.resources:
                        all_resources.extend(resources_page.resources)
                    if not resources_page.nextCursor:
                        break
                    current_cursor = resources_page.nextCursor
                resources_mapping[name] = {
                    HAS_ERROR: False,
                    "resources": all_resources,
                }
        except Exception as e:
            logger.error(f"[list_resources_async]-Error fetching resources for server {name}: {e}")
            resources_mapping[name] = {
                HAS_ERROR: True,
                ERR_INFO: str(e)
            }
    return resources_mapping


async def list_resource_templates_async(server_config: MCPServer) -> dict[str, dict[str, Any]]:
    connections = await convert_connections(server_config.mcp_servers, server_config.hosted)
    client = MultiServerMCPClient(connections)
    templates_mapping = {}
    server_name_list = server_config.query_server_names
    if server_name_list is None or len(server_name_list) == 0:
        server_name_list = server_config.mcp_servers.keys()
    for name in server_name_list:
        try:
            async with client.session(name) as session:
                current_cursor: str | None = None
                all_templates = []
                while True:
                    try:
                        templates_page = await session.list_resource_templates(cursor=current_cursor)
                    except McpError as e:
                        logger.error(f"[list_resource_templates_async]-Error fetching resource templates for server {name}: {e}")
                        break
                    if templates_page.templates:
                        all_templates.extend(templates_page.templates)
                    if not templates_page.nextCursor:
                        break
                    current_cursor = templates_page.nextCursor
                templates_mapping[name] = {
                    HAS_ERROR: False,
                    "templates": all_templates,
                }
        except Exception as e:
            logger.error(f"[list_resource_templates_async]-Error fetching resource templates for server {name}: {e}")
            templates_mapping[name] = {
                HAS_ERROR: True,
                ERR_INFO: str(e)
            }
    return templates_mapping
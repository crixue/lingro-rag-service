# from mcp import Tool
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union
from datetime import timedelta


class StdioConnection(BaseModel):
    transport: Literal["stdio"] = Field(default="stdio", alias="type")
    command: str = Field(default=None)
    args: List[str] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = Field(default=None)
    cwd: Optional[Union[str, Any]] = Field(default=None)
    encoding: Optional[str] = Field(default="utf-8")
    encoding_error_handler: Optional[Literal["strict", "ignore", "replace"]] = Field(default="strict")
    session_kwargs: Optional[Dict[str, Any]] = Field(default=None)


class SSEConnection(BaseModel):
    transport: Literal["sse"] = Field(default="sse", alias="type")
    url: str = Field(default=None)
    headers: Optional[Dict[str, Any]] = Field(default=None)
    timeout: Optional[float] = Field(default=30.0)
    sse_read_timeout: Optional[float] = Field(default=300.0, alias="sseReadTimeout")
    session_kwargs: Optional[Dict[str, Any]] = Field(default=None, alias="sessionKwargs")
    httpx_client_factory: Optional[Any] = Field(default=None)
    auth: Optional[Any] = Field(default=None)


class StreamableHttpConnection(BaseModel):
    transport: Literal["streamable_http"] = Field(default="streamable_http", alias="type")
    url: str = Field(default=None)
    headers: Optional[Dict[str, Any]] = Field(default=None)
    timeout: Optional[timedelta] = Field(default=timedelta(seconds=30))
    sse_read_timeout: Optional[timedelta] = Field(default=timedelta(seconds=300), alias="sseReadTimeout")
    terminate_on_close: Optional[bool] = Field(default=True, alias="terminateOnClose")
    session_kwargs: Optional[Dict[str, Any]] = Field(default=None, alias="sessionKwargs")
    httpx_client_factory: Optional[Any] = Field(default=None)
    auth: Optional[Any] = Field(default=None)


class WebsocketConnection(BaseModel):
    transport: Literal["websocket"] = Field(default="websocket", alias="type")
    url: str = Field(default=None)
    session_kwargs: Optional[Dict[str, Any]] = Field(default=None, alias="sessionKwargs")


# Not Support StdioConnection yet
Connection = Union[SSEConnection, StreamableHttpConnection, WebsocketConnection]


class MCPServer(BaseModel):
    mcp_servers: Dict[str, Connection] = Field(default_factory=dict, alias="mcpServers", description="A dictionary of MCP servers with their connection configurations.")
    query_server_names: Optional[List[str]] = Field(default=None, alias="queryServerNames", description="List of server names to query. If None, all servers will be queried.")
    query_server_name: Optional[str] = Field(default=None, alias="queryServerName", description="Single server name to query. If None, all servers will be queried.")
    query_prompt_name: Optional[str] = Field(default=None, alias="queryPromptName", description="Name of the prompt to query. If None, all prompts will be queried.")
    query_arguments: Optional[Dict[str, Any]] = Field(default=None, alias="queryArguments", description="Arguments to pass when querying the prompt. If None, no arguments will be passed.")
    hosted: Optional[bool] = Field(default=False, alias="hosted", description="Indicates if the MCP server is hosted. If True, the server is hosted; otherwise, it is not.")


class MCPServerWithTools(BaseModel):
    id: str = Field(default=None, alias="id", description="Unique identifier for the MCP server.")
    server_name: str = Field(default=None, alias="serverName", description="Name of the MCP server.")
    mcp_servers: Dict[str, Connection] = Field(default_factory=dict, alias="mcpServers", description="A dictionary of MCP servers with their connection configurations.")
    tools: List = Field(default_factory=list, description="List of tools available on the MCP servers.")
    hosted: Optional[bool] = Field(default=False, alias="hosted", description="Indicates if the MCP server is hosted. If True, the server is hosted; otherwise, it is not.")

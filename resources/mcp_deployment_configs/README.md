# MCP Server 配置示例

这里的 JSON 是 MCP Server 的声明示例，格式与主流 MCP 客户端一致：

```json
{
  "mcpServers": {
    "mcp/time": {
      "disabled": false,
      "timeout": 60,
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"],
      "transportType": "stdio"
    }
  }
}
```

| 文件 | Server | 传输 | 依赖 |
|---|---|---|---|
| `mcp-server-time.json` | `mcp-server-time` | stdio | `uvx` |
| `mcp-server-mcp-deepwiki.json` | `mcp-deepwiki` | stdio | `npx` |
| `mcp-server-mcp-server-chart.json` | `@antv/mcp-server-chart` | stdio | `npx` |
| `mcp-server-sequential-thinking.json` | `server-sequential-thinking` | stdio | `npx` |
| `mcp-server-12306-mcp.json` | `12306-mcp` | stdio | `npx` |
| `mcp-server-mcp-yahoo-finance.json` | `mcp-yahoo-finance` | stdio | `npx` |
| `mcp-server-mcp-trends-hub.json` | `mcp-trends-hub` | stdio | `npx` |

说明：

- 全部为公开社区 server，本仓库不包含任何凭据；
- `command` 需要在运行环境里可用（`uvx` / `npx`），首次执行会下载依赖；
- 需要认证的托管型 MCP Server（sse 传输）通过 OAuth2 client credentials 获取 access token，凭据从环境变量读取。

## 在请求中启用

对话接口的 `used_mcp_tools` 字段声明本次要用哪个 server 的哪些工具：

```json
{
  "used_mcp_tools": [
    {
      "server_name": "mcp/time",
      "hosted": false,
      "mcp_servers": { "mcp/time": { "command": "uvx", "args": ["mcp-server-time"], "transportType": "stdio" } },
      "tools": [
        { "name": "get_current_time", "description": "Get current time", "inputSchema": {"type": "object", "properties": {}} }
      ]
    }
  ]
}
```

字段定义见 `controllers/vo/mcp_connection.py`。服务端会把这些工具转换成 LangChain Tool 注入 Agent。

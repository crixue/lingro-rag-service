import unittest

from langchain_mcp_adapters.client import MultiServerMCPClient


class TestMCPTools(unittest.IsolatedAsyncioTestCase):

    async def test0(self):
        mcp_tool = {
            "time": {
                "url": "https://api.i-lingro.com/servers/mcp/time/sse",
                "transport": "sse",
            }
        }
        client = MultiServerMCPClient(
            mcp_tool
        )
        all_tools = await client.get_tools()
        print(all_tools)

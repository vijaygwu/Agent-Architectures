"""
Chapter 4: Agent Communication Protocols
MCP Client Implementation
=========================================

A client for connecting to MCP servers.

Usage:
    python mcp_client.py
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any


class MCPError(Exception):
    """Error from MCP server."""
    pass


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict

class MCPClient:
    def __init__(self):
        self.request_id = 0
        self.reader = None
        self.writer = None
        self.process = None
        self.tools: dict[str, MCPTool] = {}

    async def connect_stdio(self, command: list[str]):
        """Connect to an MCP server via stdio."""
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self.reader = self.process.stdout
        self.writer = self.process.stdin

        # Initialize
        await self._initialize()

        # Load tools
        await self._load_tools()

    async def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and wait for response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }

        self.writer.write((json.dumps(request) + "\n").encode())
        await self.writer.drain()

        line = await self.reader.readline()
        response = json.loads(line.decode())

        if "error" in response:
            raise MCPError(response["error"]["message"]) from None

        return response.get("result", {})

    async def _initialize(self):
        """Initialize the MCP connection."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "book-client", "version": "1.0.0"}
        })
        return result

    async def _load_tools(self):
        """Load available tools from server."""
        result = await self._send_request("tools/list")
        for tool_data in result.get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data["description"],
                input_schema=tool_data["inputSchema"]
            )
            self.tools[tool.name] = tool

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

        # Parse the result content
        content = result.get("content", [])
        if content and content[0]["type"] == "text":
            return json.loads(content[0]["text"])
        return result

    def get_tools_for_llm(self) -> list[dict]:
        """Get tools in format suitable for LLM function calling."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
            for tool in self.tools.values()
        ]

    async def close(self):
        """Close the MCP connection and terminate the subprocess."""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass  # Already closed or errored
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception:
                pass  # Already terminated


async def demo():
    """Demonstrate MCP client usage."""
    client = MCPClient()

    # Connect to the server (assuming it's running)
    # await client.connect_stdio(["python", "mcp_server.py"])

    # For demo, we'll simulate with direct server usage
    print("MCP Client ready")
    print(f"To connect: await client.connect_stdio(['python', 'mcp_server.py'])")

if __name__ == "__main__":
    asyncio.run(demo())

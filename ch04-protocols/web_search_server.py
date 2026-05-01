"""
Chapter 4: Agent Communication Protocols
MCP Server Implementation
=========================================

A complete MCP server providing web search and URL fetching tools.

Usage:
    python web_search_server.py
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from typing import Any

import aiohttp


# =============================================================================
# JSON-RPC Types (simplified from MCP spec)
# =============================================================================

@dataclass
class Resource:
    uri: str
    name: str
    mimeType: str = "text/plain"

@dataclass
class JsonRpcRequest:
    jsonrpc: str
    id: int | str
    method: str
    params: dict = None


@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: int | str = None
    result: Any = None
    error: dict = None


@dataclass
class Tool:
    name: str
    description: str
    inputSchema: dict


# =============================================================================
# Tool Implementations
# =============================================================================

async def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web using DuckDuckGo."""
    async with aiohttp.ClientSession() as session:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}

        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []

                # Extract related topics (with defensive checks)
                for topic in data.get("RelatedTopics", [])[:num_results]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("Text", "")[:100],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })

                return {"success": True, "results": results, "query": query}
            else:
                return {"success": False, "error": f"HTTP {response.status}"}


async def fetch_url(url: str, max_length: int = 10000) -> dict:
    """Fetch content from a URL."""
    async with aiohttp.ClientSession() as session:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    return {
                        "success": True,
                        "url": url,
                        "content": content[:max_length],
                        "truncated": len(content) > max_length,
                        "content_type": response.headers.get(
                            "Content-Type", "unknown")
                    }
                else:
                    err = f"HTTP {response.status}"
                    return {"success": False, "error": err, "url": url}
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}


# =============================================================================
# MCP Server
# =============================================================================

class MCPServer:
    def __init__(self):
        self.tools = {
            "web_search": Tool(
                name="web_search",
                description="Search the web for information using DuckDuckGo",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            "fetch_url": Tool(
                name="fetch_url",
                description="Fetch the content of a URL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum content length",
                            "default": 10000
                        }
                    },
                    "required": ["url"]
                }
            )
        }

        self.tool_handlers = {
            "web_search": web_search,
            "fetch_url": fetch_url
        }

    async def handle_request(
            self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Route JSON-RPC requests to appropriate handlers."""
        method = request.method
        params = request.params or {}

        try:
            if method == "initialize":
                return self._handle_initialize(request.id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request.id)
            elif method == "tools/call":
                return await self._handle_tools_call(request.id, params)
            elif method == "resources/list":
                return self._handle_resources_list(request.id)
            else:
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Unknown method: {method}"}
                )
        except Exception as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )

    def _handle_initialize(self, req_id: int, params: dict) -> JsonRpcResponse:
        """Handle MCP initialization."""
        return JsonRpcResponse(
            id=req_id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "book-mcp-server",
                    "version": "1.0.0"
                }
            }
        )

    def _handle_tools_list(self, req_id: int) -> JsonRpcResponse:
        """List available tools."""
        return JsonRpcResponse(
            id=req_id,
            result={
                "tools": [asdict(tool) for tool in self.tools.values()]
            }
        )

    async def _handle_tools_call(self, req_id: int, params: dict) -> JsonRpcResponse:
        """Execute a tool call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tool_handlers:
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32602, "message": f"Unknown tool: {tool_name}"}
            )

        handler = self.tool_handlers[tool_name]
        result = await handler(**arguments)

        return JsonRpcResponse(
            id=req_id,
            result={
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)}
                ]
            }
        )

    def _handle_resources_list(self, req_id: int) -> JsonRpcResponse:
        """List available resources."""
        return JsonRpcResponse(
            id=req_id,
            result={"resources": []}
        )


async def run_stdio_server():
    """Run MCP server over stdio."""
    import sys
    server = MCPServer()
    loop = asyncio.get_running_loop()

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            data = json.loads(line.decode())
            request = JsonRpcRequest(**data)
            response = await server.handle_request(request)

            response_json = json.dumps(asdict(response)) + "\n"
            writer.write(response_json.encode())
            await writer.drain()
        except json.JSONDecodeError:
            error_response = JsonRpcResponse(
                error={"code": -32700, "message": "Parse error"}
            )
            writer.write((json.dumps(asdict(error_response)) + "\n").encode())
            await writer.drain()


async def demo():
    """Demonstrate MCP server functionality."""
    server = MCPServer()

    # Initialize
    init_request = JsonRpcRequest(
        jsonrpc="2.0",
        id=1,
        method="initialize",
        params={"protocolVersion": "2024-11-05"}
    )
    init_response = await server.handle_request(init_request)
    print(f"Initialize: {init_response.result}")

    # List tools
    list_request = JsonRpcRequest(jsonrpc="2.0", id=2, method="tools/list")
    list_response = await server.handle_request(list_request)
    print(f"\nAvailable tools: {[t['name'] for t in list_response.result['tools']]}")

    # Call web_search
    search_request = JsonRpcRequest(
        jsonrpc="2.0",
        id=3,
        method="tools/call",
        params={
            "name": "web_search",
            "arguments": {"query": "Python programming", "num_results": 3}
        }
    )
    search_response = await server.handle_request(search_request)
    print(f"\nSearch results:\n{search_response.result['content'][0]['text']}")

if __name__ == "__main__":
    # Run demo if executed directly
    asyncio.run(demo())

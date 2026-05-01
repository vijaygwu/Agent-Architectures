"""
Chapter 4: Agent Communication Protocols
Integrated Agent Implementation
================================

An agent that uses MCP for tools and A2A for inter-agent communication.

Usage:
    python integrated_agent.py
"""

from typing import Any
from mcp_client import MCPClient
from a2a_client_book import A2AClient, AgentCard, TaskState


class IntegratedAgent:
    """An agent that uses MCP for tools and A2A for inter-agent communication."""

    def __init__(self, agent_card: AgentCard):
        self.card = agent_card
        self.mcp_clients: dict[str, MCPClient] = {}
        self.a2a_client = A2AClient()

    async def add_tool_server(self, name: str, command: list[str]):
        """Connect to an MCP tool server."""
        client = MCPClient()
        await client.connect_stdio(command)
        self.mcp_clients[name] = client

    async def discover_agent(self, url: str) -> AgentCard:
        """Discover another agent via A2A."""
        return await self.a2a_client.discover_agent(url)

    async def call_tool(self, server: str, tool: str, args: dict) -> Any:
        """Call a tool via MCP."""
        client = self.mcp_clients.get(server)
        if not client:
            raise ValueError(f"Unknown tool server: {server}")
        return await client.call_tool(tool, args)

    async def delegate_task(self, agent_url: str, skill: str,
                             input_data: dict) -> Any:
        """Delegate a task to another agent via A2A."""
        task = await self.a2a_client.send_task(agent_url, skill, input_data)
        completed = await self.a2a_client.wait_for_completion(agent_url, task.id)

        if completed.state == TaskState.COMPLETED:
            artifacts = await self.a2a_client.get_task_artifacts(agent_url, task.id)
            return artifacts[0].content if artifacts else None
        else:
            raise RuntimeError(f"Task failed: {completed.metadata.get('error')}")

    def get_all_tools(self) -> list[dict]:
        """Get all available tools for LLM function calling."""
        tools = []
        for name, client in self.mcp_clients.items():
            for tool in client.get_tools_for_llm():
                tool["server"] = name
                tools.append(tool)
        return tools

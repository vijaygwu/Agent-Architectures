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
    """An agent that uses MCP for tools and A2A for inter-agent communication.

    This class demonstrates protocol integration by combining:
    - MCP (Model Context Protocol) for tool access
    - A2A (Agent-to-Agent) for inter-agent communication

    Usage:
        async with IntegratedAgent(card) as agent:
            await agent.add_tool_server(...)
    """

    def __init__(self, agent_card: AgentCard):
        """Initialize the integrated agent.

        Args:
            agent_card: The A2A agent card describing this agent's capabilities.
        """
        self.card = agent_card
        self.mcp_clients: dict[str, MCPClient] = {}
        self.a2a_client = A2AClient()
        self._initialized = False

    async def __aenter__(self):
        """Initialize async resources."""
        await self.a2a_client.__aenter__()
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        await self.a2a_client.__aexit__(exc_type, exc_val, exc_tb)
        for client in self.mcp_clients.values():
            await client.close()
        self._initialized = False

    async def add_tool_server(self, name: str, command: list[str]):
        """Connect to an MCP tool server.

        Args:
            name: A unique name to identify this tool server.
            command: The command and arguments to start the MCP server process.
        """
        client = MCPClient()
        await client.connect_stdio(command)
        self.mcp_clients[name] = client

    async def discover_agent(self, url: str) -> AgentCard:
        """Discover another agent via A2A.

        Args:
            url: The base URL of the agent to discover.

        Returns:
            The discovered agent's AgentCard.
        """
        return await self.a2a_client.discover_agent(url)

    async def call_tool(self, server: str, tool: str, args: dict) -> Any:
        """Call a tool via MCP.

        Args:
            server: The name of the MCP server (as registered with add_tool_server).
            tool: The name of the tool to call.
            args: Arguments to pass to the tool.

        Returns:
            The result from the tool execution.

        Raises:
            ValueError: If the server name is not registered.
        """
        client = self.mcp_clients.get(server)
        if not client:
            raise ValueError(f"Unknown tool server: {server}")
        return await client.call_tool(tool, args)

    async def delegate_task(self, agent_url: str, skill: str,
                             input_data: dict) -> Any:
        """Delegate a task to another agent via A2A.

        Args:
            agent_url: The URL of the target agent.
            skill: The skill/capability to invoke on the target agent.
            input_data: Input data for the task.

        Returns:
            The result from the delegated task.

        Raises:
            RuntimeError: If the task fails.
        """
        task = await self.a2a_client.send_task(agent_url, skill, input_data)
        completed = await self.a2a_client.wait_for_completion(agent_url, task.id)

        if completed.state == TaskState.COMPLETED:
            artifacts = await self.a2a_client.get_task_artifacts(agent_url, task.id)
            return artifacts[0].content if artifacts else None
        else:
            raise RuntimeError(f"Task failed: {completed.metadata.get('error')}")

    def get_all_tools(self) -> list[dict]:
        """Get all available tools for LLM function calling.

        Returns:
            A list of tool definitions in a format suitable for LLM function calling,
            with each tool annotated with its source server name.
        """
        tools = []
        for name, client in self.mcp_clients.items():
            for tool in client.get_tools_for_llm():
                tool["server"] = name
                tools.append(tool)
        return tools

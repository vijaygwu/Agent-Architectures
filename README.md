# Agent Architectures

**Companion Code Repository**

*Design Patterns for Multi-Agent AI Systems*

by Vijay Raghavan

---

## About This Repository

This repository contains the complete Python implementations for all code examples in *Agent Architectures: Design Patterns for Multi-Agent AI Systems*. Each chapter's code is organized in its own directory with working, production-ready implementations.

## Book Overview

*Agent Architectures* is a comprehensive guide to designing and implementing multi-agent AI systems. The book covers four foundational patterns—Orchestrator, Council, Swarm, and Guardian—along with hybrid architectures and the infrastructure needed to support them.

## Repository Structure

```
.
├── ch04-protocols/          # MCP server and A2A client implementations
├── ch05-orchestrator/       # Orchestrator pattern with task decomposition
├── ch06-council/            # Council pattern with voting mechanisms
├── ch07-swarm/              # Swarm pattern with stigmergic coordination
├── ch08-guardian/           # Guardian pattern with policy enforcement
├── ch09-hybrid/             # Hybrid architectures combining patterns
├── ch10-identity/           # Agent identity and authentication
├── ch11-gateway/            # Policy gateway implementation
├── ch12-complete-example/   # Complete research assistant example
├── common/                  # Shared utilities and type definitions
└── requirements.txt         # Python dependencies
```

## Chapter Contents

| Chapter | Topic | Key Implementations |
|---------|-------|---------------------|
| 4 | Communication Protocols | MCP server, A2A client, message routing |
| 5 | Orchestrator Pattern | Task decomposition, worker management, result synthesis |
| 6 | Council Pattern | Multi-agent deliberation, voting mechanisms, consensus |
| 7 | Swarm Pattern | Stigmergic coordination, emergent behavior, pheromone fields |
| 8 | Guardian Pattern | Policy enforcement, action validation, security boundaries |
| 9 | Hybrid Architectures | Pattern composition, adaptive selection, fallback strategies |
| 10 | Agent Identity | JWT authentication, certificate management, identity verification |
| 11 | Policy Fundamentals | Request validation, policy enforcement, access control |
| 12 | Research Assistant | Complete example combining all patterns |

## Getting Started

### Prerequisites

- Python 3.11 or higher
- An API key for your preferred LLM provider (OpenAI, Anthropic, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/vijaygwu/Agent-Architectures.git
cd Agent-Architectures

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Running Examples

Each chapter directory contains standalone examples that can be run directly:

```bash
# Run the orchestrator example
python ch05-orchestrator/orchestrator.py

# Run the council deliberation example
python ch06-council/council.py

# Run the swarm coordination example
python ch07-swarm/swarm.py

# Run the guardian pattern example
python ch08-guardian/guardian.py

# Run the complete research assistant
python ch12-complete-example/research_agents.py
```

## Key Patterns

### Orchestrator Pattern (Chapter 5)
Central coordinator that decomposes complex tasks, assigns work to specialized agents, and synthesizes results. Best for well-defined workflows with clear task boundaries.

### Council Pattern (Chapter 6)
Multiple agents deliberate together using voting mechanisms to reach decisions on complex, ambiguous problems. Ideal for high-stakes decisions requiring diverse perspectives.

### Swarm Pattern (Chapter 7)
Many simple agents coordinate through indirect communication (stigmergy) to solve problems through emergent behavior. Effective for exploration and optimization tasks.

### Guardian Pattern (Chapter 8)
Specialized agents that monitor, validate, and enforce policies on other agents' actions. Essential for safety, compliance, and resource management.

### Hybrid Architectures (Chapter 9)
Combine multiple patterns to leverage their complementary strengths. Real-world systems often need orchestration with guardian oversight, or councils that spawn swarms.

## Part Structure

### Part I: Foundations (Chapters 1-4)
Core concepts of AI agents, the agent loop, multi-agent fundamentals, and communication protocols (MCP, A2A).

### Part II: Patterns (Chapters 5-9)
The four foundational patterns plus hybrid architectures for complex systems.

### Part III: Governance (Chapters 10-11)
Agent identity, authentication, and policy enforcement for enterprise deployments.

### Part IV: Complete Example (Chapter 12)
A production-ready research assistant demonstrating all patterns working together.

## Related Resources

- **Book**: *Agent Architectures: Design Patterns for Multi-Agent AI Systems*
- **Companion Volume**: [Agents in Production](https://github.com/vijaygwu/Agents-in-Production) - Operating Multi-Agent Systems at Scale

## License

This code is provided for educational purposes to accompany the book. See LICENSE for details.

## Author

**Vijay Raghavan**

- GitHub: [@vijaygwu](https://github.com/vijaygwu)

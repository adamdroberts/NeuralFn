# NeuralFn MCP Server

NeuralFn exposes its graph-editing, execution, and training capabilities through an MCP (Model Context Protocol) server. This allows AI agents and LLM-powered tools to interact with NeuralFn programmatically.

## Server Details

- **Framework:** FastMCP
- **Server name:** `NeuralFn Editor`
- **Entry point:** `python server/mcp_server.py` (or `uv run server/mcp_server.py`)

## Authentication

The MCP server authenticates with the NeuralFn REST API using environment variables:

| Variable | Description |
|----------|-------------|
| `NEURALFN_MCP_EMAIL` | Email of the NeuralFn user account the MCP server acts as. |
| `NEURALFN_MCP_PASSWORD` | Password for that account. |
| `NEURALFN_BASE_URL` | Base URL of the NeuralFn API. Defaults to `http://localhost:8000/api`. |

The MCP server logs in on startup and uses the resulting session cookie for all subsequent API calls.

## Scope Rules

MCP tools follow a consistent scoping convention:

| Scope | Required parameters | Examples |
|-------|---------------------|----------|
| Global | (none) | `list_builtins` |
| Project | `project_id` | `list_datasets`, `download_dataset`, `delete_dataset` |
| Session | `project_id` + `session_id` | All graph, node, edge, execution, and training tools |

Most tools operate at the session level and require both `project_id` and `session_id`. Dataset catalog tools operate at the project level. `list_builtins` is the only global tool.

## Tool Reference

- [Graph tools](graph-tools.md) -- read, replace, and configure graphs
- [Node tools](node-tools.md) -- add, update, delete, and position nodes
- [Edge tools](edge-tools.md) -- add, update, and delete edges
- [Variant tools](variant-tools.md) -- manage variant families and swap node implementations
- [Execution tools](execution-tools.md) -- forward passes, tracing, probing, templates, and training
- [Dataset tools](dataset-tools.md) -- catalog, download, load, and manage datasets

## Client Configuration Examples

### Cursor (.cursor/mcp.json)

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "server/mcp_server.py"],
      "env": {
        "NEURALFN_MCP_EMAIL": "your-email@example.com",
        "NEURALFN_MCP_PASSWORD": "your-password",
        "NEURALFN_BASE_URL": "http://localhost:8000/api"
      }
    }
  }
}
```

### Codex (.codex/config.toml)

```toml
[mcp.neuralfn]
type = "stdio"
command = ["uv", "run", "server/mcp_server.py"]

[mcp.neuralfn.env]
NEURALFN_MCP_EMAIL = "your-email@example.com"
NEURALFN_MCP_PASSWORD = "your-password"
NEURALFN_BASE_URL = "http://localhost:8000/api"
```

### Claude Desktop

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "server/mcp_server.py"],
      "cwd": "/path/to/NeuralFn",
      "env": {
        "NEURALFN_MCP_EMAIL": "your-email@example.com",
        "NEURALFN_MCP_PASSWORD": "your-password",
        "NEURALFN_BASE_URL": "http://localhost:8000/api"
      }
    }
  }
}
```

## Agent Skills

For AI-ready skill definitions that guide agents through common NeuralFn workflows, see [Agent Skills](../agent-skills.md).

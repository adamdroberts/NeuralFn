# Agent Instructions

## Documentation updates are required

Whenever you change meaningful product behavior, setup steps, routing, auth/session flows, persistence, API contracts, MCP behavior, or operational workflows, update the docs in the same task.

- Update `README.md` with the current high-level usage, setup, and workflow guidance a user or developer needs immediately.
- Append a more detailed entry to `CHANGELOG.md` describing what changed, important implementation or migration notes, and how the change was verified.

## When this applies

Treat the documentation update requirement as mandatory for:

- user-facing feature or workflow changes
- backend or frontend setup/run instruction changes
- authentication, project/session, routing, dataset, or training workflow changes
- REST API or MCP contract changes
- operational or environment-variable changes

## Done criteria

A meaningful feature change is not complete until the relevant `README.md` and `CHANGELOG.md` updates are included.

# NeuralFn REST API

## Base URL

All endpoints are served under `/api`.

## Authentication

NeuralFn uses HTTP-only session cookies for authentication. On successful login or admin bootstrap, the server sets a `neuralfn_session` cookie. Most endpoints enforce authentication through the `require_auth` dependency -- unauthenticated requests receive a `401` response.

A few endpoints (notably `GET /api/bootstrap`) use optional authentication: they succeed without a cookie but return richer data when one is present.

## Content Type

All request and response bodies use `application/json` unless noted otherwise:

- **File uploads** (`POST /api/projects/{project_id}/datasets/upload`) use `multipart/form-data`.
- **Training progress** (`POST .../runs`) streams `text/event-stream` (Server-Sent Events).

## Error Format

Errors are returned as JSON objects with a `detail` field. The value is either a plain string or a structured dictionary with additional context.

```json
{
  "detail": "Session not found"
}
```

### Status Codes

| Code | Meaning |
|------|---------|
| 400  | Bad request -- invalid input, missing fields, or constraint violation |
| 401  | Unauthorized -- missing or expired session cookie |
| 403  | Forbidden -- authenticated but lacking permission (e.g. non-admin) |
| 404  | Not found -- resource does not exist or is not accessible |
| 409  | Conflict -- revision mismatch on graph update |
| 500  | Internal server error |

### Revision Conflicts (409)

`PUT .../graph` uses optimistic concurrency. If the `expected_revision` in the request body does not match the server's current revision, the endpoint returns `409` with:

```json
{
  "message": "Revision conflict",
  "current_revision": 42,
  "graph": { "...current server-side graph..." }
}
```

The client should merge or discard local changes using the returned graph and retry with the updated revision.

## API Reference

- [Bootstrap](bootstrap.md) -- initial app state
- [Auth](auth.md) -- login, logout, admin setup, session switching
- [Admin](admin.md) -- user and membership management
- [Projects](projects.md) -- project CRUD and analytics
- [Sessions](sessions.md) -- sessions, graphs, nodes, edges, execution, templates, datasets, agent
- [Datasets](datasets.md) -- dataset catalog (download, upload, access control)
- [Training Runs](runs.md) -- start, monitor, and stop training runs

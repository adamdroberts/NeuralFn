# Projects

Endpoints for managing projects. A project is the top-level organizational unit that contains sessions, graphs, and dataset access grants.

---

## GET /api/projects

Lists all projects accessible to the authenticated user.

**Authentication:** required.

### Response

```json
[
  {
    "id": "p_xyz",
    "name": "My Project",
    "description": "Language model experiments"
  },
  {
    "id": "p_abc",
    "name": "Vision",
    "description": null
  }
]
```

---

## POST /api/projects

Creates a new project. The authenticated user is automatically added as the owner.

**Authentication:** required.

### Request Body

```json
{
  "name": "New Project",
  "description": "Optional description"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Project name. |
| `description` | string | no | Optional description. |

### Response

Returns the new project and its automatically created default session.

```json
{
  "project": {
    "id": "p_new1",
    "name": "New Project",
    "description": "Optional description"
  },
  "default_session": {
    "id": "s_001",
    "name": "default",
    "project_id": "p_new1"
  }
}
```

---

## GET /api/projects/{project_id}

Returns details for a single project.

**Authentication:** required. The user must be a member of the project.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `project_id` | The project to retrieve. |

### Response

```json
{
  "id": "p_xyz",
  "name": "My Project",
  "description": "Language model experiments"
}
```

Returns `404` if the project does not exist or the user lacks access.

---

## GET /api/projects/{project_id}/analytics

Returns an analytics summary for the project, including aggregate metrics across sessions and training runs.

**Authentication:** required. The user must be a member of the project.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `project_id` | The project to query. |

### Response

The response shape depends on the available data. A typical response includes session counts, run statistics, and dataset usage.

```json
{
  "total_sessions": 5,
  "total_runs": 23,
  "total_datasets": 3,
  "recent_runs": [
    {
      "id": "r_001",
      "session_id": "s_001",
      "status": "completed",
      "final_loss": 0.042
    }
  ]
}
```

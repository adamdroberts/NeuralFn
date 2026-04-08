# Admin

Administrative endpoints for managing users and project memberships. All endpoints in this section require the caller to be an admin (`is_admin: true`).

All endpoints are prefixed with `/api/admin`.

---

## GET /api/admin/users

Returns all registered users.

**Authentication:** required (admin only).

### Response

```json
[
  {
    "id": "u_abc123",
    "email": "admin@example.com",
    "display_name": "Admin",
    "is_admin": true
  },
  {
    "id": "u_def456",
    "email": "user@example.com",
    "display_name": "Researcher",
    "is_admin": false
  }
]
```

---

## POST /api/admin/users

Creates a new user account.

**Authentication:** required (admin only).

### Request Body

```json
{
  "email": "user@example.com",
  "password": "hunter2",
  "display_name": "Researcher",
  "is_admin": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `email` | string | yes | Must be unique across all users. |
| `password` | string | yes | Plaintext password (hashed server-side). |
| `display_name` | string | yes | Human-readable name. |
| `is_admin` | bool | yes | Whether the new user has admin privileges. |

### Response

Returns the created user object.

```json
{
  "id": "u_ghi789",
  "email": "user@example.com",
  "display_name": "Researcher",
  "is_admin": false
}
```

Returns `409` if the email is already taken.

---

## GET /api/admin/projects/{project_id}/memberships

Lists all memberships for a project.

**Authentication:** required (admin only).

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `project_id` | The project to query. |

### Response

```json
[
  {
    "id": "m_001",
    "project_id": "p_xyz",
    "user_id": "u_abc123",
    "role": "owner"
  },
  {
    "id": "m_002",
    "project_id": "p_xyz",
    "user_id": "u_def456",
    "role": "editor"
  }
]
```

---

## POST /api/admin/projects/{project_id}/memberships

Adds a user to a project. Specify the user by `user_id` or `email` (at least one is required).

**Authentication:** required (admin only).

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `project_id` | The project to add the member to. |

### Request Body

```json
{
  "user_id": "u_def456",
  "role": "editor"
}
```

Or by email:

```json
{
  "email": "user@example.com",
  "role": "viewer"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | no | User ID. Provide this or `email`. |
| `email` | string | no | User email. Provide this or `user_id`. |
| `role` | string | yes | Role to assign (e.g. `"owner"`, `"editor"`, `"viewer"`). |

### Response

Returns the created membership object.

```json
{
  "id": "m_003",
  "project_id": "p_xyz",
  "user_id": "u_def456",
  "role": "editor"
}
```

Returns `404` if the user is not found. Returns `409` if the user is already a member of the project.

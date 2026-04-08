# Bootstrap

## GET /api/bootstrap

Returns the current application state in a single round-trip. This is the first call a client should make on page load.

**Authentication:** not required (optional auth). When a valid session cookie is present the response includes user-specific data; otherwise only setup status is returned.

### Response

```json
{
  "requires_setup": false,
  "authenticated": true,
  "user": {
    "id": "u_abc123",
    "email": "admin@example.com",
    "display_name": "Admin",
    "is_admin": true
  },
  "projects": [
    {
      "id": "p_xyz",
      "name": "My Project",
      "description": "..."
    }
  ],
  "active_project_id": "p_xyz",
  "active_session_id": "s_001",
  "active_session": {
    "id": "s_001",
    "name": "default",
    "project_id": "p_xyz"
  }
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `requires_setup` | bool | `true` when no admin user exists yet. The client should show the bootstrap-admin form. |
| `authenticated` | bool | `true` when the request carried a valid session cookie. |
| `user` | object or null | The authenticated user, or `null` if not logged in. |
| `projects` | array | Projects the user has access to. Empty array when unauthenticated. |
| `active_project_id` | string or null | The project currently selected in the user's server-side session state. |
| `active_session_id` | string or null | The session (within the active project) currently selected. |
| `active_session` | object or null | Full session object for the active session. |

### Notes

- When `requires_setup` is `true`, the only valid next action is `POST /api/auth/bootstrap-admin`.
- When `authenticated` is `false` but `requires_setup` is also `false`, the client should redirect to the login form.

# Auth

Authentication endpoints manage user identity and the server-side active session pointer.

All endpoints that set or clear the session cookie use the `neuralfn_session` HTTP-only cookie.

---

## POST /api/auth/bootstrap-admin

Creates the first admin user, a default project, and a default session. This endpoint is only available when no users exist in the database.

**Authentication:** none (endpoint is disabled after first use).

### Request Body

```json
{
  "email": "admin@example.com",
  "password": "s3cret",
  "display_name": "Admin"
}
```

### Response

Returns the created user object and sets the session cookie.

```json
{
  "user": {
    "id": "u_abc123",
    "email": "admin@example.com",
    "display_name": "Admin",
    "is_admin": true
  }
}
```

---

## POST /api/auth/login

Authenticates an existing user by email and password.

**Authentication:** none.

### Request Body

```json
{
  "email": "admin@example.com",
  "password": "s3cret"
}
```

### Response

```json
{
  "user": {
    "id": "u_abc123",
    "email": "admin@example.com",
    "display_name": "Admin",
    "is_admin": true
  }
}
```

Sets the `neuralfn_session` cookie on success. Returns `401` on invalid credentials.

---

## POST /api/auth/logout

Clears the session cookie.

**Authentication:** required.

### Response

```json
{
  "status": "logged_out"
}
```

---

## GET /api/auth/me

Returns the currently authenticated user.

**Authentication:** required.

### Response

```json
{
  "id": "u_abc123",
  "email": "admin@example.com",
  "display_name": "Admin",
  "is_admin": true
}
```

---

## PUT /api/auth/active-session

Switches the user's active project and optionally the active session within that project. If `session_id` is omitted the server selects the project's default session.

**Authentication:** required.

### Request Body

```json
{
  "project_id": "p_xyz",
  "session_id": "s_002"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | string | yes | Target project ID. |
| `session_id` | string | no | Target session within the project. If omitted, the server picks the default. |

### Response

```json
{
  "project_id": "p_xyz",
  "session_id": "s_002"
}
```

# Authentication

## Password and Token Utilities

`server/security.py` provides low-level auth primitives:

- **`hash_password(plain)`** -- returns a bcrypt hash of the plaintext password.
- **`verify_password(plain, hashed)`** -- checks a plaintext password against a bcrypt hash.
- **`new_session_token()`** -- generates a cryptographically random session token string.
- **`hash_session_token(token)`** -- returns the SHA-256 hex digest of a session token (stored in the database; the raw token is never persisted).

## AuthService

`server/services/auth_service.py` contains all authentication and user-management logic.

### Exceptions

- **`AuthError`** -- raised for invalid credentials, expired sessions, or missing accounts.
- **`PermissionDenied`** -- raised when an authenticated user lacks the required role or privilege.

### AuthContext

```python
@dataclass
class AuthContext:
    user: User
    auth_session: AuthSession
```

Returned by `authenticate_token()` and threaded through request handlers via FastAPI dependencies.

### Methods

| Method | Description |
|--------|-------------|
| `setup_required()` | Returns `True` if no admin user exists yet (first-run state). |
| `list_users()` | Returns all registered users. |
| `get_user_by_email(email)` | Looks up a user by email address. |
| `create_user(email, password, display_name, is_admin)` | Creates a new user account. |
| `bootstrap_admin(email, password, display_name)` | Creates the initial admin user. Only succeeds when no users exist. |
| `login(email, password)` | Verifies credentials and returns a new session token. |
| `logout(token)` | Invalidates the session associated with the given token. |
| `authenticate_token(token)` | Validates a session token and returns an `AuthContext`. Raises `AuthError` if expired or invalid. |
| `set_active_scope(auth_session, project_id, session_id)` | Updates the session's active project and editor session pointers. |

## FastAPI Dependencies

`server/dependencies.py` defines injectable dependencies:

- **`get_db()`** -- yields a database session (calls `get_db_session()`).
- **`get_optional_auth()`** -- extracts the session cookie and returns an `AuthContext` or `None`. Does not raise on missing/invalid tokens.
- **`require_auth()`** -- same as `get_optional_auth()`, but raises HTTP 401 if authentication fails.

Route handlers declare their auth requirements by adding `auth: AuthContext = Depends(require_auth)` to their signatures.

## Cookie Flow

1. **Login**: the client POSTs credentials to `/api/auth/login`. On success, the server sets an HTTP-only cookie named by `settings.session_cookie_name` containing the raw session token.
2. **Subsequent requests**: the browser sends the cookie automatically. `require_auth()` reads the cookie, calls `authenticate_token()`, and injects the resulting `AuthContext`.
3. **Logout**: the client POSTs to `/api/auth/logout`. The server deletes the cookie and removes the `AuthSession` row.
4. **Expiry**: sessions expire after `session_ttl_seconds`. Expired sessions are rejected by `authenticate_token()`.

## Native inference Bearer authentication

The standalone `nfn infer --serve` application does not mount or reuse the
editor cookie flow above. Loopback binding is unauthenticated unless an
operator configures `NFN_INFER_API_KEY` or `--api-key-file`. Once configured,
every inference route, including `/health`, requires `Authorization: Bearer
<key>` and compares the supplied value in constant time.

Key files may contain multiple non-empty lines for rotation, but must not be
group/world accessible. Supplying both the environment variable and key file
is rejected. Non-loopback binding requires one of those key sources unless
`--allow-unauthenticated-remote` explicitly accepts unauthenticated exposure.
Invalid or missing credentials return the OpenAI error envelope with HTTP 401,
`code: "invalid_api_key"`, and `WWW-Authenticate: Bearer`.

The native server does not create editor users, cookie sessions, or records in
the editor database. When `--state-db` is configured, it writes Responses,
Conversations, compaction references, and background-job records only to its
separate private SQLite store, partitioned by the accepted Bearer-key
fingerprint. See
[the native inference serving contract](../rest-api/native-inference-serving.md).

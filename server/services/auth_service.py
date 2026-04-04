from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db_models import AuthSession, User, ensure_utc, utcnow
from ..security import hash_password, hash_session_token, new_session_token, verify_password
from ..settings import get_settings


class AuthError(ValueError):
    pass


class PermissionDenied(ValueError):
    pass


@dataclass
class AuthContext:
    user: User
    auth_session: AuthSession


class AuthService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def setup_required(self, db: Session) -> bool:
        return db.scalar(select(func.count(User.id))) == 0

    def list_users(self, db: Session) -> list[User]:
        return list(db.scalars(select(User).order_by(User.created_at.asc())))

    def get_user_by_email(self, db: Session, email: str) -> User | None:
        return db.scalar(select(User).where(User.email == email.lower().strip()))

    def create_user(
        self,
        db: Session,
        *,
        email: str,
        password: str,
        display_name: str,
        is_admin: bool,
    ) -> User:
        normalized_email = email.lower().strip()
        if not normalized_email:
            raise AuthError("Email is required")
        if len(password) < 4:
            raise AuthError("Password must be at least 4 characters")
        if self.get_user_by_email(db, normalized_email):
            raise AuthError("A user with that email already exists")
        user = User(
            email=normalized_email,
            display_name=display_name.strip() or normalized_email,
            password_hash=hash_password(password),
            is_admin=is_admin,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def bootstrap_admin(self, db: Session, *, email: str, password: str, display_name: str) -> User:
        if not self.setup_required(db):
            raise AuthError("Bootstrap has already completed")
        return self.create_user(
            db,
            email=email,
            password=password,
            display_name=display_name,
            is_admin=True,
        )

    def login(self, db: Session, *, email: str, password: str) -> tuple[User, str, AuthSession]:
        user = self.get_user_by_email(db, email)
        if user is None or not verify_password(password, user.password_hash):
            raise AuthError("Invalid email or password")

        raw_token = new_session_token()
        auth_session = AuthSession(
            token_hash=hash_session_token(raw_token),
            user_id=user.id,
            last_seen_at=utcnow(),
            expires_at=utcnow() + timedelta(seconds=self._settings.session_ttl_seconds),
        )
        db.add(auth_session)
        db.commit()
        db.refresh(auth_session)
        return user, raw_token, auth_session

    def logout(self, db: Session, raw_token: str) -> None:
        token_hash = hash_session_token(raw_token)
        auth_session = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_hash))
        if auth_session is None:
            return
        db.delete(auth_session)
        db.commit()

    def authenticate_token(self, db: Session, raw_token: str) -> AuthContext | None:
        token_hash = hash_session_token(raw_token)
        auth_session = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_hash))
        if auth_session is None:
            return None
        expires_at = ensure_utc(auth_session.expires_at)
        if expires_at is not None and expires_at < utcnow():
            db.delete(auth_session)
            db.commit()
            return None
        user = db.get(User, auth_session.user_id)
        if user is None:
            return None
        now = utcnow()
        auth_session.last_seen_at = now
        auth_session.expires_at = now + timedelta(seconds=self._settings.session_ttl_seconds)
        db.commit()
        db.refresh(auth_session)
        return AuthContext(user=user, auth_session=auth_session)

    def set_active_scope(
        self,
        db: Session,
        auth_session: AuthSession,
        *,
        project_id: str | None,
        session_id: str | None,
    ) -> AuthSession:
        auth_session.current_project_id = project_id
        auth_session.current_editor_session_id = session_id
        auth_session.last_seen_at = utcnow()
        db.commit()
        db.refresh(auth_session)
        return auth_session


_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service

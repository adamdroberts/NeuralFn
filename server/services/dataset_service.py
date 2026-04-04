from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..dataset_manager import (
    delete_dataset as delete_local_dataset,
    download_hf_dataset,
    get_local_dataset_info,
    list_local_datasets,
    load_dataset_tokens,
    upload_local_file,
)
from ..db_models import DatasetAsset, Project, ProjectDatasetGrant, User
from .auth_service import PermissionDenied
from .session_service import get_workspace_service


class DatasetService:
    def __init__(self) -> None:
        self._workspace = get_workspace_service()

    def _accessible_project_ids(self, db: Session, user: User) -> set[str]:
        return {project["id"] for project in self._workspace.list_projects(db, user)}

    def _normalize_project_ids(self, project_id: str, requested_ids: Iterable[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in [project_id, *(requested_ids or [])]:
            value = str(candidate or "").strip()
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    def _require_project_ids(self, project_ids: list[str]) -> None:
        if not project_ids:
            raise PermissionDenied("Select at least one project for dataset access")

    def _ensure_manageable_projects(self, db: Session, user: User, project_ids: list[str]) -> None:
        accessible_ids = self._accessible_project_ids(db, user)
        invalid = [project_id for project_id in project_ids if project_id not in accessible_ids]
        if invalid:
            raise PermissionDenied("You do not have access to one or more selected projects")

    def _apply_dataset_info(self, asset: DatasetAsset, info: dict[str, Any], *, created_by_user_id: str | None) -> None:
        asset.source = str(info.get("source") or "local")
        asset.hf_path = info.get("hf_path")
        asset.hf_split = info.get("hf_split")
        asset.text_column = str(info.get("text_column") or "text")
        asset.num_tokens = int(info["num_tokens"]) if info.get("num_tokens") is not None else None
        asset.num_rows = int(info["num_rows"]) if info.get("num_rows") is not None else None
        asset.variant = info.get("variant")
        asset.train_shards = int(info["train_shards"]) if info.get("train_shards") is not None else None
        asset.val_shards = int(info["val_shards"]) if info.get("val_shards") is not None else None
        asset.data_format = info.get("data_format")
        asset.repo_id = info.get("repo_id")
        asset.remote_root_prefix = info.get("remote_root_prefix")
        if created_by_user_id is not None:
            asset.created_by_user_id = created_by_user_id

    def _sync_grants(self, db: Session, asset: DatasetAsset, project_ids: list[str]) -> None:
        existing = list(
            db.scalars(
                select(ProjectDatasetGrant).where(ProjectDatasetGrant.dataset_id == asset.id)
            )
        )
        keep = set(project_ids)
        for grant in existing:
            if grant.project_id not in keep:
                db.delete(grant)
        existing_project_ids = {grant.project_id for grant in existing}
        for project_id in project_ids:
            if project_id not in existing_project_ids:
                db.add(ProjectDatasetGrant(project_id=project_id, dataset_id=asset.id))

    def _grant_map(self, db: Session, dataset_ids: list[str]) -> dict[str, list[str]]:
        if not dataset_ids:
            return {}
        rows = db.execute(
            select(ProjectDatasetGrant.dataset_id, ProjectDatasetGrant.project_id).where(
                ProjectDatasetGrant.dataset_id.in_(dataset_ids)
            )
        )
        grants: dict[str, list[str]] = {}
        for dataset_id, project_id in rows:
            grants.setdefault(dataset_id, []).append(project_id)
        for project_ids in grants.values():
            project_ids.sort()
        return grants

    def _serialize_asset(self, asset: DatasetAsset, project_ids: list[str]) -> dict[str, Any]:
        return {
            "name": asset.name,
            "source": asset.source,
            "hf_path": asset.hf_path,
            "hf_split": asset.hf_split,
            "text_column": asset.text_column,
            "num_tokens": asset.num_tokens,
            "num_rows": asset.num_rows,
            "variant": asset.variant,
            "train_shards": asset.train_shards,
            "val_shards": asset.val_shards,
            "data_format": asset.data_format,
            "repo_id": asset.repo_id,
            "remote_root_prefix": asset.remote_root_prefix,
            "project_ids": project_ids,
        }

    def reconcile_local_catalog(self, db: Session) -> None:
        local_datasets = {item["name"]: item for item in list_local_datasets()}
        assets = list(db.scalars(select(DatasetAsset).order_by(DatasetAsset.name.asc())))
        assets_by_name = {asset.name: asset for asset in assets}
        project_ids = list(db.scalars(select(Project.id).order_by(Project.created_at.asc())))
        mutated = False

        for asset in assets:
            info = local_datasets.get(asset.name)
            if info is None:
                db.delete(asset)
                mutated = True
                continue
            before = self._serialize_asset(asset, [])
            self._apply_dataset_info(asset, info, created_by_user_id=asset.created_by_user_id)
            if self._serialize_asset(asset, []) != before:
                mutated = True
            grant_count = len(
                list(
                    db.scalars(
                        select(ProjectDatasetGrant.id).where(ProjectDatasetGrant.dataset_id == asset.id)
                    )
                )
            )
            if grant_count == 0 and project_ids:
                self._sync_grants(db, asset, project_ids)
                mutated = True

        for name, info in local_datasets.items():
            if name in assets_by_name:
                continue
            asset = DatasetAsset(name=name)
            self._apply_dataset_info(asset, info, created_by_user_id=None)
            db.add(asset)
            db.flush()
            if project_ids:
                self._sync_grants(db, asset, project_ids)
            mutated = True

        if mutated:
            db.commit()

    def _get_asset(self, db: Session, dataset_name: str) -> DatasetAsset | None:
        return db.scalar(select(DatasetAsset).where(DatasetAsset.name == dataset_name))

    def _get_visible_asset(self, db: Session, project_id: str, dataset_name: str) -> DatasetAsset | None:
        return db.scalar(
            select(DatasetAsset)
            .join(ProjectDatasetGrant, ProjectDatasetGrant.dataset_id == DatasetAsset.id)
            .where(
                DatasetAsset.name == dataset_name,
                ProjectDatasetGrant.project_id == project_id,
            )
        )

    def list_datasets(self, db: Session, user: User, project_id: str) -> list[dict[str, Any]]:
        self._workspace.get_project(db, user, project_id)
        self.reconcile_local_catalog(db)
        assets = list(
            db.scalars(
                select(DatasetAsset)
                .join(ProjectDatasetGrant, ProjectDatasetGrant.dataset_id == DatasetAsset.id)
                .where(ProjectDatasetGrant.project_id == project_id)
                .order_by(DatasetAsset.name.asc())
            )
        )
        grant_map = self._grant_map(db, [asset.id for asset in assets])
        return [self._serialize_asset(asset, grant_map.get(asset.id, [])) for asset in assets]

    def ensure_dataset_access(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        dataset_names: Iterable[str],
    ) -> list[str]:
        self._workspace.get_project(db, user, project_id)
        self.reconcile_local_catalog(db)
        names = [str(name).strip() for name in dataset_names if str(name).strip()]
        if not names:
            return []
        visible = set(
            db.scalars(
                select(DatasetAsset.name)
                .join(ProjectDatasetGrant, ProjectDatasetGrant.dataset_id == DatasetAsset.id)
                .where(
                    ProjectDatasetGrant.project_id == project_id,
                    DatasetAsset.name.in_(names),
                )
            )
        )
        missing = [name for name in names if name not in visible]
        if missing:
            raise PermissionDenied(f"Dataset access denied or not found: {', '.join(missing)}")
        return names

    def load_dataset_tokens_for_project(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        dataset_names: Iterable[str],
        seq_len: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        names = self.ensure_dataset_access(db, user, project_id=project_id, dataset_names=dataset_names)
        return load_dataset_tokens(names, seq_len=seq_len)

    def _upsert_asset(
        self,
        db: Session,
        *,
        dataset_info: dict[str, Any],
        project_ids: list[str],
        created_by_user_id: str | None,
    ) -> dict[str, Any]:
        asset = self._get_asset(db, dataset_info["name"])
        if asset is None:
            asset = DatasetAsset(name=dataset_info["name"])
            db.add(asset)
            db.flush()
        self._apply_dataset_info(asset, dataset_info, created_by_user_id=created_by_user_id)
        self._sync_grants(db, asset, project_ids)
        db.commit()
        db.refresh(asset)
        grant_map = self._grant_map(db, [asset.id])
        return self._serialize_asset(asset, grant_map.get(asset.id, []))

    def _existing_project_ids(self, db: Session, dataset_name: str) -> list[str]:
        asset = self._get_asset(db, dataset_name)
        if asset is None:
            return []
        return self._grant_map(db, [asset.id]).get(asset.id, [])

    def download_dataset(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        request,
    ) -> dict[str, Any]:
        self._workspace.get_project(db, user, project_id)
        dataset_info = download_hf_dataset(
            request.hf_path,
            hf_split=request.hf_split,
            text_column=request.text_column,
            max_rows=request.max_rows,
            alias=request.alias,
            variant=request.variant,
            train_shards=request.train_shards,
            skip_manifest=request.skip_manifest,
            with_docs=request.with_docs,
            repo_id=request.repo_id,
            remote_root_prefix=request.remote_root_prefix,
        )
        requested_project_ids = getattr(request, "project_ids", None)
        project_ids = self._normalize_project_ids(
            project_id,
            requested_project_ids if requested_project_ids is not None else self._existing_project_ids(db, dataset_info["name"]),
        )
        self._require_project_ids(project_ids)
        if requested_project_ids is not None:
            self._ensure_manageable_projects(db, user, project_ids)
        return self._upsert_asset(
            db,
            dataset_info=dataset_info,
            project_ids=project_ids,
            created_by_user_id=user.id,
        )

    def upload_dataset(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        name: str,
        content: bytes,
        filename: str,
        project_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        self._workspace.get_project(db, user, project_id)
        normalized_project_ids = self._normalize_project_ids(
            project_id,
            project_ids if project_ids is not None else self._existing_project_ids(db, name),
        )
        self._require_project_ids(normalized_project_ids)
        if project_ids is not None:
            self._ensure_manageable_projects(db, user, normalized_project_ids)
        dataset_info = upload_local_file(name, content, filename)
        return self._upsert_asset(
            db,
            dataset_info=dataset_info,
            project_ids=normalized_project_ids,
            created_by_user_id=user.id,
        )

    def update_dataset_access(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        dataset_name: str,
        project_ids: Iterable[str],
    ) -> dict[str, Any]:
        self._workspace.get_project(db, user, project_id)
        normalized_project_ids = self._normalize_project_ids(project_id, project_ids)
        self._require_project_ids(normalized_project_ids)
        self._ensure_manageable_projects(db, user, normalized_project_ids)
        self.reconcile_local_catalog(db)
        asset = self._get_visible_asset(db, project_id, dataset_name)
        if asset is None:
            raise PermissionDenied("Dataset not found")
        self._sync_grants(db, asset, normalized_project_ids)
        db.commit()
        db.refresh(asset)
        grant_map = self._grant_map(db, [asset.id])
        return self._serialize_asset(asset, grant_map.get(asset.id, []))

    def delete_dataset(self, db: Session, user: User, *, project_id: str, dataset_name: str) -> dict[str, Any]:
        self._workspace.get_project(db, user, project_id)
        self.reconcile_local_catalog(db)
        asset = self._get_visible_asset(db, project_id, dataset_name)
        if asset is None:
            raise PermissionDenied("Dataset not found")
        delete_local_dataset(asset.name)
        db.delete(asset)
        db.commit()
        return {"status": "deleted"}

    def dataset_info(self, db: Session, dataset_name: str) -> dict[str, Any] | None:
        self.reconcile_local_catalog(db)
        asset = self._get_asset(db, dataset_name)
        if asset is None:
            return get_local_dataset_info(dataset_name)
        grant_map = self._grant_map(db, [asset.id])
        return self._serialize_asset(asset, grant_map.get(asset.id, []))


_dataset_service: DatasetService | None = None


def get_dataset_service() -> DatasetService:
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service

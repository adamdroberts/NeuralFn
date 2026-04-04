from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..dependencies import get_db, require_auth
from ..models import DatasetAccessUpdateRequest, DownloadDatasetRequest
from ..services.auth_service import AuthContext, PermissionDenied
from ..services.dataset_service import get_dataset_service

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])


@router.get("")
def get_datasets(
    project_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list[dict]:
    try:
        return get_dataset_service().list_datasets(db, auth.user, project_id)
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/download")
def download_dataset(
    project_id: str,
    body: DownloadDatasetRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_dataset_service().download_dataset(
            db,
            auth.user,
            project_id=project_id,
            request=body,
        )
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload")
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    name: str = Form(...),
    project_ids: str | None = Form(None),
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        content = await file.read()
        parsed_project_ids = json.loads(project_ids) if project_ids else None
        if parsed_project_ids is not None and not isinstance(parsed_project_ids, list):
            raise ValueError("project_ids must be a JSON array")
        return get_dataset_service().upload_dataset(
            db,
            auth.user,
            project_id=project_id,
            name=name,
            content=content,
            filename=file.filename or "data.txt",
            project_ids=parsed_project_ids,
        )
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/{ds_name}/access")
def update_dataset_access(
    project_id: str,
    ds_name: str,
    body: DatasetAccessUpdateRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_dataset_service().update_dataset_access(
            db,
            auth.user,
            project_id=project_id,
            dataset_name=ds_name,
            project_ids=body.project_ids,
        )
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{ds_name}")
def remove_dataset(
    project_id: str,
    ds_name: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_dataset_service().delete_dataset(
            db,
            auth.user,
            project_id=project_id,
            dataset_name=ds_name,
        )
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

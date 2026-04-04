from __future__ import annotations

from fastapi import APIRouter, Depends

from neuralfn.builtins import BuiltinNeurons

from ..dependencies import require_auth
from ..models import GPTTemplateRequest
from ..services.graph_ops import build_template_payload

router = APIRouter(tags=["catalog"])


@router.get("/builtins")
def list_builtins(_auth=Depends(require_auth)) -> list[dict]:
    return [builtin.to_dict() for builtin in BuiltinNeurons.all()]


@router.post("/templates/gpt")
def build_gpt_template(body: GPTTemplateRequest, _auth=Depends(require_auth)) -> dict:
    return build_template_payload(body)

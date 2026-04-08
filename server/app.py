from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routes import router
from .settings import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from .services.persistence_worker import get_persistence_worker
    init_db()
    get_persistence_worker().start()
    yield
    get_persistence_worker().stop()


settings = get_settings()
app = FastAPI(title="NeuralFn", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

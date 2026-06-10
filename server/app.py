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

# Mount static files and handle catch-all SPA routing when built in production
import os
from pathlib import Path

dist_dir = Path(__file__).resolve().parent.parent / "editor" / "dist"
if dist_dir.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # Mount static assets directory
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Catch-all handler for React Router HTML5 History mode (BrowserRouter)
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        # Let API routes handle themselves, if they reached here, they don't exist
        if path.startswith("api"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="API route not found")
            
        index_file = dist_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Frontend index.html not found")


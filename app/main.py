"""FastAPI application entrypoint."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes.qa import router as qa_router
from app.routes.recommend import router as recommend_router
from app.routes.wardrobe import router as wardrobe_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "app" / "static"
DEMO_IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "demo_images"
WARDROBE_IMAGE_ROOT = PROJECT_ROOT / "data" / "user_wardrobe"

app = FastAPI(title="Fashion Bot API")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
app.mount("/demo_images", StaticFiles(directory=DEMO_IMAGE_ROOT), name="demo_images")
app.mount("/user_wardrobe", StaticFiles(directory=WARDROBE_IMAGE_ROOT), name="user_wardrobe")
app.include_router(qa_router, prefix="/qa", tags=["qa"])
app.include_router(recommend_router, prefix="/recommend", tags=["recommend"])
app.include_router(wardrobe_router, prefix="/wardrobe", tags=["wardrobe"])


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

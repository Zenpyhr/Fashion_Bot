"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.routes.qa import router as qa_router
from app.routes.recommend import router as recommend_router

app = FastAPI(title="Fashion Bot API")
app.include_router(qa_router, prefix="/qa", tags=["qa"])
app.include_router(recommend_router, prefix="/recommend", tags=["recommend"])


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

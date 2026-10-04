"""
FastAPI application entrypoint.

React frontend -> FastAPI REST API -> Service layer -> ML/Prediction layer
-> Database / Model storage.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, dashboard, predict, predictions, reports
from app.config import get_settings
from app.database.session import init_db
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="Ensemble ML + Explainable AI credit card fraud detection system (academic mini project).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces / internals to the client.
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:  # noqa: BLE001
        logger.error("Database initialization failed: %s", e)
        logger.error("Check DATABASE_URL in your .env and ensure PostgreSQL is running.")


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.ENV}


app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(predict.router)
app.include_router(predictions.router)
app.include_router(reports.router)

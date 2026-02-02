"""
FlashAI REST API Server

Provides HTTP endpoints for:
- Reasoning queries
- Learning operations
- State management (save/reset)
- GitHub webhooks
- System status
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from flashai.core.engine import FlashAIEngine
from flashai.core.config import FlashAIConfig
from flashai.github.webhooks import GitHubWebhookHandler


logger = logging.getLogger(__name__)


# Request/Response Models
class ReasonRequest(BaseModel):
    """Request model for reasoning endpoint."""
    query: str = Field(..., description="The query to reason about")
    context: Optional[dict[str, Any]] = Field(None, description="Optional context")
    constraints: Optional[list[str]] = Field(None, description="Constraints to satisfy")


class LearnRequest(BaseModel):
    """Request model for learning endpoint."""
    examples: list[dict[str, Any]] = Field(..., description="Training examples")
    feedback: Optional[dict[str, Any]] = Field(None, description="Optional feedback")


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""
    export_name: Optional[str] = Field(None, description="Name for export file")
    include_user_data: bool = Field(True, description="Include user data in export")


class RestoreRequest(BaseModel):
    """Request model for restore endpoint."""
    export_path: str = Field(..., description="Path to export zip file")


class InitializeRequest(BaseModel):
    """Request model for initialization endpoint."""
    user_id: Optional[str] = Field(None, description="Optional user identifier")


# Global engine instance
_engine: Optional[FlashAIEngine] = None
_webhook_handler: Optional[GitHubWebhookHandler] = None


def get_engine() -> FlashAIEngine:
    """Get the global engine instance."""
    global _engine
    if _engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return _engine


def create_app(
    base_path: Optional[Path] = None,
    config: Optional[FlashAIConfig] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        base_path: Base path for FlashAI installation
        config: Optional configuration override

    Returns:
        Configured FastAPI application
    """
    global _engine, _webhook_handler

    app = FastAPI(
        title="FlashAI API",
        description="Portable AI System with Energy-Based Reasoning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Initialize engine
    _engine = FlashAIEngine(base_path=base_path, config=config)

    # Initialize webhook handler
    async def on_learn_trigger(data: dict[str, Any]) -> None:
        """Callback for webhook-triggered learning."""
        await _engine.learn({"examples": [], "source_data": data})

    _webhook_handler = GitHubWebhookHandler(
        config=_engine.config.github,
        on_learn_trigger=on_learn_trigger,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_engine.config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Startup/shutdown events
    @app.on_event("startup")
    async def startup():
        logger.info("FlashAI API server starting...")

    @app.on_event("shutdown")
    async def shutdown():
        if _engine:
            await _engine.shutdown()
        logger.info("FlashAI API server stopped")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "flashai"}

    # System endpoints
    @app.post("/api/v1/initialize")
    async def initialize(request: InitializeRequest):
        """Initialize the FlashAI system for a user."""
        engine = get_engine()
        result = await engine.initialize(user_identifier=request.user_id)
        return result

    @app.get("/api/v1/status")
    async def get_status():
        """Get system status."""
        engine = get_engine()
        return await engine.get_status()

    # Reasoning endpoints
    @app.post("/api/v1/reason")
    async def reason(request: ReasonRequest):
        """Perform energy-based reasoning on a query."""
        engine = get_engine()
        try:
            result = await engine.reason(
                query=request.query,
                context=request.context,
                constraints=request.constraints,
            )
            return result
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Learning endpoints
    @app.post("/api/v1/learn")
    async def learn(request: LearnRequest):
        """Learn from provided examples."""
        engine = get_engine()
        try:
            result = await engine.learn(
                data={"examples": request.examples},
                feedback=request.feedback,
            )
            return result
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # State management endpoints
    @app.post("/api/v1/save-and-reset")
    async def save_and_reset(request: ResetRequest):
        """Save current state and reset to initial state."""
        engine = get_engine()
        result = await engine.save_and_reset(
            export_name=request.export_name,
            include_user_data=request.include_user_data,
        )
        return result

    @app.post("/api/v1/restore")
    async def restore(request: RestoreRequest):
        """Restore from an export file."""
        engine = get_engine()
        try:
            result = await engine.restore_from_export(request.export_path)
            return result
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/v1/exports")
    async def list_exports():
        """List available export files."""
        engine = get_engine()
        exports = await engine.state_manager.list_exports()
        return {"exports": exports}

    @app.get("/api/v1/exports/{export_id}")
    async def download_export(export_id: str):
        """Download an export file."""
        engine = get_engine()
        export_path = engine.state_manager.exports_path / f"{export_id}.zip"
        if not export_path.exists():
            raise HTTPException(status_code=404, detail="Export not found")
        return FileResponse(
            path=export_path,
            filename=f"{export_id}.zip",
            media_type="application/zip",
        )

    @app.get("/api/v1/checkpoints")
    async def list_checkpoints():
        """List available checkpoints."""
        engine = get_engine()
        checkpoints = await engine.state_manager.list_checkpoints()
        return {"checkpoints": checkpoints}

    @app.post("/api/v1/checkpoints/{checkpoint_id}/restore")
    async def restore_checkpoint(checkpoint_id: str):
        """Restore from a checkpoint."""
        engine = get_engine()
        try:
            result = await engine.state_manager.restore_checkpoint(checkpoint_id)
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # GitHub webhook endpoint
    @app.post("/webhooks/github")
    async def github_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_github_event: str = Header(..., alias="X-GitHub-Event"),
        x_github_delivery: str = Header(..., alias="X-GitHub-Delivery"),
        x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
    ):
        """Handle incoming GitHub webhooks."""
        if _webhook_handler is None:
            raise HTTPException(status_code=500, detail="Webhook handler not initialized")

        raw_payload = await request.body()
        payload = await request.json()

        result = await _webhook_handler.handle_webhook(
            event_type=x_github_event,
            payload=payload,
            delivery_id=x_github_delivery,
            signature=x_hub_signature_256,
            raw_payload=raw_payload,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)

        return {
            "status": "ok",
            "message": result.message,
            "actions": result.actions_taken,
        }

    # Model information endpoints
    @app.get("/api/v1/model/info")
    async def get_model_info():
        """Get information about the current model."""
        engine = get_engine()
        return {
            "model_type": "EnergyBasedReasoningModel",
            "config": engine.config.ebm.model_dump(),
            "device": engine.config.resolve_device(),
            "loaded": engine._ebm_model is not None,
        }

    @app.get("/api/v1/model/energy-breakdown")
    async def get_energy_breakdown(query: str, answer: str):
        """Get energy breakdown for a query-answer pair."""
        engine = get_engine()
        breakdown = engine.ebm_model.get_energy_breakdown(query, answer)
        return {"breakdown": breakdown}

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8420,
    base_path: Optional[Path] = None,
    reload: bool = False,
) -> None:
    """
    Run the FlashAI API server.

    Args:
        host: Server host
        port: Server port
        base_path: Base path for FlashAI installation
        reload: Enable auto-reload for development
    """
    import uvicorn

    # Create app with base path
    app = create_app(base_path=base_path)

    logger.info(f"Starting FlashAI server at http://{host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()

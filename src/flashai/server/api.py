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

from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flashai.core.engine import FlashAIEngine
from flashai.core.config import FlashAIConfig
from flashai.github.webhooks import GitHubWebhookHandler
from flashai.learning.file_parser import LearningDataParser


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

    # File upload endpoints
    @app.post("/api/v1/learn/upload")
    async def upload_learning_data(
        file: UploadFile = File(...),
        apply_immediately: bool = Form(True),
    ):
        """
        Upload a file containing learning data.

        Supported formats:
        - JSON: Array of examples or object with 'examples' key
        - JSONL: JSON Lines format (one JSON object per line)
        - CSV: Columns for query/question and answer/response
        - YAML: YAML format with examples list
        - Markdown: Headers as queries, content as answers
        - TXT: Q:/A: patterns or line pairs
        """
        engine = get_engine()

        # Read file content
        content = await file.read()

        # Parse the file
        parser = LearningDataParser()
        result = parser.parse_upload(
            file_content=content,
            filename=file.filename or "upload",
            content_type=file.content_type,
        )

        if not result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Failed to parse file",
                    "errors": result.errors,
                    "format_detected": result.format_detected.value,
                },
            )

        response = {
            "status": "parsed",
            "filename": file.filename,
            "format": result.format_detected.value,
            "examples_found": len(result.examples),
            "warnings": result.warnings,
        }

        # Apply learning if requested
        if apply_immediately:
            try:
                learning_data = result.to_learning_data()
                learn_result = await engine.learn(data=learning_data)
                response["status"] = "learned"
                response["learning_result"] = learn_result
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        return response

    @app.get("/api/v1/learn/formats")
    async def get_supported_formats():
        """Get list of supported file formats for learning data upload."""
        parser = LearningDataParser()
        return {"formats": parser.get_supported_formats()}

    @app.post("/api/v1/learn/preview")
    async def preview_learning_data(file: UploadFile = File(...)):
        """Preview parsed learning data without applying it."""
        content = await file.read()

        parser = LearningDataParser()
        result = parser.parse_upload(
            file_content=content,
            filename=file.filename or "upload",
            content_type=file.content_type,
        )

        return {
            "success": result.success,
            "format": result.format_detected.value,
            "examples_count": len(result.examples),
            "examples_preview": [ex.to_dict() for ex in result.examples[:5]],
            "errors": result.errors,
            "warnings": result.warnings,
        }

    # History endpoints
    @app.get("/api/v1/history")
    async def get_interaction_history(
        limit: int = 50,
        offset: int = 0,
        interaction_type: Optional[str] = None,
    ):
        """Get interaction history for the current user."""
        engine = get_engine()
        if not engine._current_user_id:
            raise HTTPException(status_code=400, detail="No user session active")

        history = await engine.user_manager.get_interaction_history(
            user_id=engine._current_user_id,
            limit=limit,
            offset=offset,
            interaction_type=interaction_type,
        )
        return {"history": history, "total": len(history)}

    @app.delete("/api/v1/history")
    async def clear_history():
        """Clear interaction history for the current user."""
        engine = get_engine()
        if not engine._current_user_id:
            raise HTTPException(status_code=400, detail="No user session active")

        await engine.user_manager.clear_history(engine._current_user_id)
        return {"status": "cleared"}

    # Settings endpoints
    @app.get("/api/v1/settings")
    async def get_user_settings():
        """Get current user settings."""
        engine = get_engine()
        if not engine._current_user_id:
            return {"settings": engine.config.model_dump()}

        profile = await engine.user_manager.get_profile(engine._current_user_id)
        if profile:
            return {
                "settings": profile.settings,
                "preferences": profile.learned_preferences,
            }
        return {"settings": {}}

    @app.put("/api/v1/settings")
    async def update_user_settings(settings: dict[str, Any]):
        """Update user settings."""
        engine = get_engine()
        if not engine._current_user_id:
            raise HTTPException(status_code=400, detail="No user session active")

        await engine.user_manager.update_settings(
            user_id=engine._current_user_id,
            settings=settings,
        )
        return {"status": "updated", "settings": settings}

    # Notes endpoints
    @app.get("/api/v1/notes")
    async def get_notes(tag: Optional[str] = None):
        """Get user notes."""
        engine = get_engine()
        if not engine._current_user_id:
            raise HTTPException(status_code=400, detail="No user session active")

        notes = await engine.user_manager.get_notes(
            user_id=engine._current_user_id,
            tag=tag,
        )
        return {"notes": notes}

    @app.post("/api/v1/notes")
    async def create_note(
        content: str = Form(...),
        title: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),
    ):
        """Create a new note."""
        engine = get_engine()
        if not engine._current_user_id:
            raise HTTPException(status_code=400, detail="No user session active")

        tag_list = tags.split(",") if tags else []
        note = await engine.user_manager.create_note(
            user_id=engine._current_user_id,
            content=content,
            title=title,
            tags=tag_list,
        )
        return {"status": "created", "note": note}

    @app.delete("/api/v1/notes/{note_id}")
    async def delete_note(note_id: str):
        """Delete a note."""
        engine = get_engine()
        if not engine._current_user_id:
            raise HTTPException(status_code=400, detail="No user session active")

        await engine.user_manager.delete_note(
            user_id=engine._current_user_id,
            note_id=note_id,
        )
        return {"status": "deleted"}

    # Screen capture learning endpoint
    @app.post("/api/v1/learn/screen")
    async def learn_from_screen_capture(
        image: UploadFile = File(...),
        context: Optional[str] = Form(None),
    ):
        """Learn from a screen capture image."""
        engine = get_engine()

        content = await image.read()
        context_dict = json.loads(context) if context else {}

        result = await engine.learn_from_screen(
            image_data=content,
            context=context_dict,
        )

        return result

    # Web UI route
    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        """Serve the main web UI."""
        return get_ui_html()

    return app


def get_ui_html() -> str:
    """Return the main UI HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlashAI - Portable AI System</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #22d3ee;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #475569;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }

        .app-container {
            display: flex;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 260px;
            background: var(--bg-card);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }

        .logo {
            padding: 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .logo-text {
            font-size: 20px;
            font-weight: 700;
        }

        .nav-menu {
            flex: 1;
            padding: 16px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 4px;
            color: var(--text-muted);
        }

        .nav-item:hover, .nav-item.active {
            background: var(--primary);
            color: var(--text);
        }

        .nav-item svg {
            width: 20px;
            height: 20px;
        }

        .status-bar {
            padding: 16px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-muted);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
        }

        .status-dot.offline {
            background: var(--error);
        }

        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-title {
            font-size: 18px;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            gap: 12px;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: var(--primary);
            color: var(--text);
        }

        .btn-primary:hover {
            background: var(--primary-dark);
        }

        .btn-secondary {
            background: var(--bg-input);
            color: var(--text);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--border);
        }

        .content-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        /* Chat Interface */
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding-bottom: 24px;
        }

        .message {
            margin-bottom: 16px;
            display: flex;
            gap: 12px;
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .message-avatar.user {
            background: var(--primary);
        }

        .message-avatar.ai {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
        }

        .message-content {
            flex: 1;
            background: var(--bg-card);
            padding: 16px;
            border-radius: 12px;
            line-height: 1.6;
        }

        .message-meta {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 8px;
            display: flex;
            gap: 16px;
        }

        .input-area {
            padding: 16px;
            background: var(--bg-card);
            border-radius: 16px;
            margin-top: auto;
        }

        .input-row {
            display: flex;
            gap: 12px;
        }

        .input-field {
            flex: 1;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
            color: var(--text);
            font-size: 15px;
            resize: none;
        }

        .input-field:focus {
            outline: none;
            border-color: var(--primary);
        }

        .input-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }

        .icon-btn {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            border: none;
            background: var(--bg-input);
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .icon-btn:hover {
            background: var(--border);
            color: var(--text);
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Settings Grid */
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 16px;
        }

        .setting-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }

        .setting-label {
            font-size: 14px;
        }

        .setting-description {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Toggle Switch */
        .toggle {
            width: 48px;
            height: 26px;
            background: var(--bg-input);
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: all 0.2s;
        }

        .toggle.active {
            background: var(--primary);
        }

        .toggle::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: var(--text);
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: all 0.2s;
        }

        .toggle.active::after {
            left: 24px;
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 48px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
        }

        .upload-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }

        /* History Table */
        .history-table {
            width: 100%;
            border-collapse: collapse;
        }

        .history-table th,
        .history-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .history-table th {
            color: var(--text-muted);
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
        }

        /* Notes */
        .note-card {
            background: var(--bg-input);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }

        .note-title {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .note-tags {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }

        .tag {
            padding: 4px 12px;
            background: var(--primary);
            border-radius: 20px;
            font-size: 12px;
        }

        /* Energy Display */
        .energy-display {
            display: flex;
            gap: 24px;
            margin: 16px 0;
        }

        .energy-item {
            flex: 1;
            text-align: center;
        }

        .energy-value {
            font-size: 32px;
            font-weight: 700;
            color: var(--primary);
        }

        .energy-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Panels */
        .panel {
            display: none;
        }

        .panel.active {
            display: block;
        }

        /* Hidden file input */
        .file-input {
            display: none;
        }

        /* Toast notifications */
        .toast-container {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 1000;
        }

        .toast {
            background: var(--bg-card);
            border-left: 4px solid var(--success);
            padding: 16px 24px;
            border-radius: 8px;
            margin-top: 12px;
            animation: slideIn 0.3s ease;
        }

        .toast.error {
            border-color: var(--error);
        }

        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <nav class="sidebar">
            <div class="logo">
                <div class="logo-icon">⚡</div>
                <span class="logo-text">FlashAI</span>
            </div>

            <div class="nav-menu">
                <div class="nav-item active" data-panel="chat">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
                    Chat
                </div>
                <div class="nav-item" data-panel="learn">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
                    Learn
                </div>
                <div class="nav-item" data-panel="history">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    History
                </div>
                <div class="nav-item" data-panel="notes">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
                    Notes
                </div>
                <div class="nav-item" data-panel="settings">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                    Settings
                </div>
            </div>

            <div class="status-bar">
                <div class="status-indicator">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">Connected</span>
                </div>
                <div>Portable Mode: <strong id="portableMode">Active</strong></div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Chat Panel -->
            <div class="panel active" id="chat-panel">
                <div class="header">
                    <h1 class="header-title">Energy-Based Reasoning</h1>
                    <div class="header-actions">
                        <button class="btn btn-secondary" onclick="clearChat()">Clear Chat</button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="chat-container">
                        <div class="messages" id="messages">
                            <div class="message">
                                <div class="message-avatar ai">⚡</div>
                                <div class="message-content">
                                    Welcome to FlashAI! I use energy-based reasoning to help you think through problems.
                                    <br><br>
                                    You can ask me questions, upload learning data, or explore the settings to customize your experience.
                                    <div class="message-meta">
                                        <span>Energy: 0.12</span>
                                        <span>Confidence: 94%</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="input-area">
                            <div class="input-row">
                                <textarea class="input-field" id="queryInput" rows="2" placeholder="Ask me anything..."></textarea>
                                <button class="btn btn-primary" onclick="sendQuery()" style="height: auto;">
                                    <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
                                </button>
                            </div>
                            <div class="input-actions">
                                <button class="icon-btn" onclick="document.getElementById('chatFileInput').click()" title="Upload file">
                                    <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"/></svg>
                                </button>
                                <input type="file" id="chatFileInput" class="file-input" onchange="handleFileUpload(this.files[0])">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Learn Panel -->
            <div class="panel" id="learn-panel">
                <div class="header">
                    <h1 class="header-title">Upload Learning Data</h1>
                </div>
                <div class="content-area">
                    <div class="card">
                        <h2 class="card-title">📁 Upload Training Data</h2>
                        <div class="upload-area" onclick="document.getElementById('learnFileInput').click()">
                            <div class="upload-icon">📄</div>
                            <h3>Drag & drop files here</h3>
                            <p style="color: var(--text-muted); margin-top: 8px;">or click to browse</p>
                            <p style="color: var(--text-muted); font-size: 12px; margin-top: 16px;">
                                Supports: JSON, JSONL, CSV, YAML, Markdown, TXT
                            </p>
                        </div>
                        <input type="file" id="learnFileInput" class="file-input" accept=".json,.jsonl,.csv,.yaml,.yml,.md,.txt" onchange="uploadLearningData(this.files[0])">
                    </div>

                    <div class="card">
                        <h2 class="card-title">📚 Supported Formats</h2>
                        <div id="formatsContainer">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- History Panel -->
            <div class="panel" id="history-panel">
                <div class="header">
                    <h1 class="header-title">Interaction History</h1>
                    <div class="header-actions">
                        <button class="btn btn-secondary" onclick="clearHistory()">Clear History</button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="card">
                        <table class="history-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Type</th>
                                    <th>Query</th>
                                    <th>Energy</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No history yet</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Notes Panel -->
            <div class="panel" id="notes-panel">
                <div class="header">
                    <h1 class="header-title">Notes & Memory</h1>
                    <div class="header-actions">
                        <button class="btn btn-primary" onclick="showNewNoteForm()">+ New Note</button>
                    </div>
                </div>
                <div class="content-area">
                    <div id="newNoteForm" class="card" style="display: none;">
                        <h2 class="card-title">Create Note</h2>
                        <input type="text" id="noteTitle" class="input-field" placeholder="Note title..." style="margin-bottom: 12px; width: 100%;">
                        <textarea id="noteContent" class="input-field" rows="4" placeholder="Note content..." style="width: 100%;"></textarea>
                        <input type="text" id="noteTags" class="input-field" placeholder="Tags (comma-separated)" style="margin-top: 12px; width: 100%;">
                        <div style="margin-top: 16px; display: flex; gap: 12px;">
                            <button class="btn btn-primary" onclick="saveNote()">Save Note</button>
                            <button class="btn btn-secondary" onclick="hideNewNoteForm()">Cancel</button>
                        </div>
                    </div>
                    <div id="notesContainer">
                        <p style="color: var(--text-muted);">No notes yet. Create one to get started!</p>
                    </div>
                </div>
            </div>

            <!-- Settings Panel -->
            <div class="panel" id="settings-panel">
                <div class="header">
                    <h1 class="header-title">Settings & Personalization</h1>
                    <div class="header-actions">
                        <button class="btn btn-primary" onclick="saveSettings()">Save Changes</button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="settings-grid">
                        <div class="card">
                            <h2 class="card-title">🎨 Appearance</h2>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Dark Mode</div>
                                    <div class="setting-description">Use dark theme</div>
                                </div>
                                <div class="toggle active" data-setting="dark_mode"></div>
                            </div>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Compact View</div>
                                    <div class="setting-description">Reduce spacing in chat</div>
                                </div>
                                <div class="toggle" data-setting="compact_view"></div>
                            </div>
                        </div>

                        <div class="card">
                            <h2 class="card-title">🧠 Learning</h2>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Auto-Learn</div>
                                    <div class="setting-description">Learn from interactions automatically</div>
                                </div>
                                <div class="toggle active" data-setting="auto_learn"></div>
                            </div>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Screen Learning</div>
                                    <div class="setting-description">Learn from screen captures when enabled</div>
                                </div>
                                <div class="toggle" data-setting="screen_learning"></div>
                            </div>
                        </div>

                        <div class="card">
                            <h2 class="card-title">⚡ Energy Model</h2>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Reasoning Depth</div>
                                    <div class="setting-description">Max reasoning steps</div>
                                </div>
                                <input type="number" class="input-field" value="10" min="1" max="50" style="width: 80px;" data-setting="reasoning_depth">
                            </div>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Confidence Threshold</div>
                                    <div class="setting-description">Minimum confidence for answers</div>
                                </div>
                                <input type="number" class="input-field" value="0.7" min="0" max="1" step="0.1" style="width: 80px;" data-setting="confidence_threshold">
                            </div>
                        </div>

                        <div class="card">
                            <h2 class="card-title">💾 Data & Privacy</h2>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Save History</div>
                                    <div class="setting-description">Store interaction history</div>
                                </div>
                                <div class="toggle active" data-setting="save_history"></div>
                            </div>
                            <div class="setting-item">
                                <div>
                                    <div class="setting-label">Encrypt Data</div>
                                    <div class="setting-description">Encrypt stored data</div>
                                </div>
                                <div class="toggle active" data-setting="encrypt_data"></div>
                            </div>
                        </div>
                    </div>

                    <div class="card" style="margin-top: 24px;">
                        <h2 class="card-title">🔄 Reset Options</h2>
                        <p style="color: var(--text-muted); margin-bottom: 16px;">
                            Save your current data and reset the AI to its initial state.
                        </p>
                        <button class="btn btn-secondary" onclick="saveAndReset()" style="background: var(--warning); border-color: var(--warning);">
                            Save & Reset
                        </button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <div class="toast-container" id="toastContainer"></div>

    <script>
        // API base URL
        const API_BASE = '';

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initializeNavigation();
            initializeToggles();
            checkStatus();
            loadFormats();
            loadHistory();
            loadNotes();

            // Handle Enter key in query input
            document.getElementById('queryInput').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendQuery();
                }
            });
        });

        // Navigation
        function initializeNavigation() {
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', () => {
                    const panel = item.dataset.panel;

                    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

                    item.classList.add('active');
                    document.getElementById(panel + '-panel').classList.add('active');
                });
            });
        }

        // Toggle switches
        function initializeToggles() {
            document.querySelectorAll('.toggle').forEach(toggle => {
                toggle.addEventListener('click', () => {
                    toggle.classList.toggle('active');
                });
            });
        }

        // Check system status
        async function checkStatus() {
            try {
                const response = await fetch(API_BASE + '/api/v1/status');
                const data = await response.json();

                document.getElementById('statusDot').classList.remove('offline');
                document.getElementById('statusText').textContent = 'Connected';
                document.getElementById('portableMode').textContent = data.base_path?.includes('flash') ? 'Active' : 'Inactive';
            } catch (error) {
                document.getElementById('statusDot').classList.add('offline');
                document.getElementById('statusText').textContent = 'Offline';
            }
        }

        // Send reasoning query
        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;

            // Add user message
            addMessage(query, 'user');
            input.value = '';

            try {
                // First initialize if needed
                await fetch(API_BASE + '/api/v1/initialize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });

                const response = await fetch(API_BASE + '/api/v1/reason', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query})
                });

                const data = await response.json();

                const answer = data.chain?.final_answer || 'Reasoning complete.';
                const energy = data.final_energy?.toFixed(4) || 'N/A';
                const confidence = ((data.confidence || 0) * 100).toFixed(0);

                addMessage(answer, 'ai', {energy, confidence});

            } catch (error) {
                addMessage('Error: ' + error.message, 'ai');
                showToast('Error processing request', 'error');
            }
        }

        // Add message to chat
        function addMessage(content, type, meta = {}) {
            const messages = document.getElementById('messages');
            const message = document.createElement('div');
            message.className = 'message';

            const avatar = type === 'user' ? '👤' : '⚡';
            const metaHtml = type === 'ai' && meta.energy ? `
                <div class="message-meta">
                    <span>Energy: ${meta.energy}</span>
                    <span>Confidence: ${meta.confidence}%</span>
                </div>
            ` : '';

            message.innerHTML = `
                <div class="message-avatar ${type}">${avatar}</div>
                <div class="message-content">
                    ${content}
                    ${metaHtml}
                </div>
            `;

            messages.appendChild(message);
            messages.scrollTop = messages.scrollHeight;
        }

        // Clear chat
        function clearChat() {
            document.getElementById('messages').innerHTML = '';
        }

        // File upload handling
        async function handleFileUpload(file) {
            if (!file) return;
            showToast('Processing file: ' + file.name);
            await uploadLearningData(file);
        }

        // Upload learning data
        async function uploadLearningData(file) {
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);
            formData.append('apply_immediately', 'true');

            try {
                const response = await fetch(API_BASE + '/api/v1/learn/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.status === 'learned') {
                    showToast(`Learned from ${data.examples_found} examples!`);
                } else if (data.status === 'parsed') {
                    showToast(`Parsed ${data.examples_found} examples`);
                } else {
                    showToast('Upload failed: ' + JSON.stringify(data.errors), 'error');
                }
            } catch (error) {
                showToast('Upload error: ' + error.message, 'error');
            }
        }

        // Load supported formats
        async function loadFormats() {
            try {
                const response = await fetch(API_BASE + '/api/v1/learn/formats');
                const data = await response.json();

                const container = document.getElementById('formatsContainer');
                container.innerHTML = data.formats.map(f => `
                    <div style="margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--border);">
                        <strong>${f.format.toUpperCase()}</strong> (${f.extensions.join(', ')})
                        <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">${f.description}</p>
                        <code style="display: block; margin-top: 8px; padding: 8px; background: var(--bg-input); border-radius: 4px; font-size: 12px;">${f.example}</code>
                    </div>
                `).join('');
            } catch (error) {
                document.getElementById('formatsContainer').innerHTML = '<p>Could not load formats</p>';
            }
        }

        // Load history
        async function loadHistory() {
            try {
                const response = await fetch(API_BASE + '/api/v1/history');
                const data = await response.json();

                const tbody = document.getElementById('historyTableBody');
                if (data.history && data.history.length > 0) {
                    tbody.innerHTML = data.history.map(h => `
                        <tr>
                            <td>${new Date(h.timestamp).toLocaleString()}</td>
                            <td>${h.type}</td>
                            <td>${h.query?.substring(0, 50)}...</td>
                            <td>${h.energy?.toFixed(4) || 'N/A'}</td>
                        </tr>
                    `).join('');
                }
            } catch (error) {
                // History not available
            }
        }

        // Clear history
        async function clearHistory() {
            if (!confirm('Clear all history?')) return;

            try {
                await fetch(API_BASE + '/api/v1/history', {method: 'DELETE'});
                loadHistory();
                showToast('History cleared');
            } catch (error) {
                showToast('Error clearing history', 'error');
            }
        }

        // Notes functions
        function showNewNoteForm() {
            document.getElementById('newNoteForm').style.display = 'block';
        }

        function hideNewNoteForm() {
            document.getElementById('newNoteForm').style.display = 'none';
        }

        async function saveNote() {
            const formData = new FormData();
            formData.append('title', document.getElementById('noteTitle').value);
            formData.append('content', document.getElementById('noteContent').value);
            formData.append('tags', document.getElementById('noteTags').value);

            try {
                await fetch(API_BASE + '/api/v1/notes', {
                    method: 'POST',
                    body: formData
                });

                hideNewNoteForm();
                loadNotes();
                showToast('Note saved!');
            } catch (error) {
                showToast('Error saving note', 'error');
            }
        }

        async function loadNotes() {
            try {
                const response = await fetch(API_BASE + '/api/v1/notes');
                const data = await response.json();

                const container = document.getElementById('notesContainer');
                if (data.notes && data.notes.length > 0) {
                    container.innerHTML = data.notes.map(n => `
                        <div class="note-card">
                            <div class="note-title">${n.title || 'Untitled'}</div>
                            <p>${n.content}</p>
                            ${n.tags?.length ? `<div class="note-tags">${n.tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>` : ''}
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<p style="color: var(--text-muted);">No notes yet. Create one to get started!</p>';
                }
            } catch (error) {
                // Notes not available
            }
        }

        // Settings
        async function saveSettings() {
            const settings = {};

            document.querySelectorAll('.toggle').forEach(toggle => {
                settings[toggle.dataset.setting] = toggle.classList.contains('active');
            });

            document.querySelectorAll('input[data-setting]').forEach(input => {
                settings[input.dataset.setting] = input.type === 'number' ? parseFloat(input.value) : input.value;
            });

            try {
                await fetch(API_BASE + '/api/v1/settings', {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(settings)
                });
                showToast('Settings saved!');
            } catch (error) {
                showToast('Error saving settings', 'error');
            }
        }

        // Save and reset
        async function saveAndReset() {
            if (!confirm('This will save your current data and reset the AI. Continue?')) return;

            try {
                const response = await fetch(API_BASE + '/api/v1/save-and-reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });

                const data = await response.json();
                showToast('Reset complete! Export saved to: ' + data.export.path);
            } catch (error) {
                showToast('Error during reset', 'error');
            }
        }

        // Toast notification
        function showToast(message, type = 'success') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast' + (type === 'error' ? ' error' : '');
            toast.textContent = message;
            container.appendChild(toast);

            setTimeout(() => toast.remove(), 5000);
        }
    </script>
</body>
</html>'''


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

"""
GitHub Webhook Handler for FlashAI.

Handles incoming webhook events from GitHub and triggers
appropriate learning and response actions.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from flashai.core.config import GitHubConfig


logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Supported GitHub webhook event types."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_REVIEW = "pull_request_review"
    ISSUES = "issues"
    ISSUE_COMMENT = "issue_comment"
    CREATE = "create"
    DELETE = "delete"
    RELEASE = "release"
    WORKFLOW_RUN = "workflow_run"
    REPOSITORY_DISPATCH = "repository_dispatch"
    PING = "ping"


@dataclass
class WebhookEvent:
    """Parsed webhook event."""
    event_type: WebhookEventType
    action: Optional[str]
    sender: str
    repository: str
    payload: dict[str, Any]
    delivery_id: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "action": self.action,
            "sender": self.sender,
            "repository": self.repository,
            "delivery_id": self.delivery_id,
            "timestamp": self.timestamp,
        }


@dataclass
class WebhookResponse:
    """Response from webhook handler."""
    success: bool
    message: str
    actions_taken: list[str] = field(default_factory=list)
    learning_triggered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class GitHubWebhookHandler:
    """
    Handles GitHub webhook events for FlashAI.

    Responsibilities:
    - Verify webhook signatures
    - Parse and validate events
    - Route events to appropriate handlers
    - Trigger learning from repository changes
    """

    def __init__(
        self,
        config: GitHubConfig,
        on_learn_trigger: Optional[Callable] = None,
    ):
        self.config = config
        self.on_learn_trigger = on_learn_trigger

        # Event handlers registry
        self._handlers: dict[WebhookEventType, list[Callable]] = {}

        # Register default handlers
        self._register_default_handlers()

        logger.info("GitHubWebhookHandler initialized")

    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self.register_handler(WebhookEventType.PUSH, self._handle_push)
        self.register_handler(WebhookEventType.PULL_REQUEST, self._handle_pull_request)
        self.register_handler(WebhookEventType.ISSUES, self._handle_issues)
        self.register_handler(WebhookEventType.ISSUE_COMMENT, self._handle_issue_comment)
        self.register_handler(WebhookEventType.PING, self._handle_ping)

    def register_handler(
        self,
        event_type: WebhookEventType,
        handler: Callable,
    ) -> None:
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Raw request body
            signature: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid
        """
        if not self.config.webhook_secret:
            logger.warning("No webhook secret configured, skipping verification")
            return True

        if not signature or not signature.startswith("sha256="):
            return False

        expected_sig = "sha256=" + hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected_sig)

    def parse_event(
        self,
        event_type_str: str,
        payload: dict[str, Any],
        delivery_id: str,
    ) -> WebhookEvent:
        """Parse webhook payload into event object."""
        from datetime import datetime

        # Map string to enum
        try:
            event_type = WebhookEventType(event_type_str)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type_str}")
            event_type = WebhookEventType.REPOSITORY_DISPATCH

        # Extract common fields
        sender = payload.get("sender", {}).get("login", "unknown")
        repository = payload.get("repository", {}).get("full_name", "unknown")
        action = payload.get("action")

        return WebhookEvent(
            event_type=event_type,
            action=action,
            sender=sender,
            repository=repository,
            payload=payload,
            delivery_id=delivery_id,
            timestamp=datetime.utcnow().isoformat(),
        )

    async def handle_webhook(
        self,
        event_type: str,
        payload: dict[str, Any],
        delivery_id: str,
        signature: Optional[str] = None,
        raw_payload: Optional[bytes] = None,
    ) -> WebhookResponse:
        """
        Handle incoming webhook event.

        Args:
            event_type: X-GitHub-Event header value
            payload: Parsed JSON payload
            delivery_id: X-GitHub-Delivery header value
            signature: X-Hub-Signature-256 header (optional)
            raw_payload: Raw request body for signature verification

        Returns:
            WebhookResponse with handling results
        """
        # Verify signature if provided
        if signature and raw_payload:
            if not self.verify_signature(raw_payload, signature):
                logger.warning(f"Invalid webhook signature for delivery {delivery_id}")
                return WebhookResponse(
                    success=False,
                    message="Invalid signature",
                )

        # Check if event type is configured
        if event_type not in self.config.webhook_events:
            logger.debug(f"Ignoring unconfigured event type: {event_type}")
            return WebhookResponse(
                success=True,
                message=f"Event type {event_type} ignored",
            )

        # Parse event
        event = self.parse_event(event_type, payload, delivery_id)
        logger.info(f"Handling webhook: {event.event_type.value} from {event.sender}")

        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            return WebhookResponse(
                success=True,
                message=f"No handlers for {event.event_type.value}",
            )

        # Execute handlers
        actions_taken = []
        learning_triggered = False

        for handler in handlers:
            try:
                result = await handler(event)
                if result:
                    actions_taken.append(result.get("action", "handler_executed"))
                    if result.get("learning_triggered"):
                        learning_triggered = True
            except Exception as e:
                logger.error(f"Handler error for {event.event_type.value}: {e}")

        return WebhookResponse(
            success=True,
            message=f"Processed {event.event_type.value}",
            actions_taken=actions_taken,
            learning_triggered=learning_triggered,
            metadata={"event": event.to_dict()},
        )

    async def _handle_ping(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle ping events (webhook setup verification)."""
        logger.info(f"Received ping from {event.repository}")
        return {
            "action": "ping_acknowledged",
            "zen": event.payload.get("zen", ""),
        }

    async def _handle_push(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle push events - trigger learning from code changes."""
        commits = event.payload.get("commits", [])
        ref = event.payload.get("ref", "")
        branch = ref.replace("refs/heads/", "")

        logger.info(f"Push to {branch}: {len(commits)} commits")

        # Trigger learning if enabled
        if self.config.enable_auto_learn and self.on_learn_trigger:
            learning_data = {
                "source": "github_push",
                "branch": branch,
                "commits": [
                    {
                        "sha": c.get("id", "")[:8],
                        "message": c.get("message", ""),
                        "author": c.get("author", {}).get("name", ""),
                        "files_modified": len(c.get("modified", [])),
                        "files_added": len(c.get("added", [])),
                        "files_removed": len(c.get("removed", [])),
                    }
                    for c in commits
                ],
            }

            try:
                await self.on_learn_trigger(learning_data)
                return {
                    "action": "push_processed",
                    "learning_triggered": True,
                    "commits_count": len(commits),
                }
            except Exception as e:
                logger.error(f"Learning trigger failed: {e}")

        return {
            "action": "push_logged",
            "commits_count": len(commits),
        }

    async def _handle_pull_request(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle pull request events."""
        action = event.action
        pr = event.payload.get("pull_request", {})
        pr_number = pr.get("number")
        pr_title = pr.get("title", "")

        logger.info(f"PR #{pr_number} {action}: {pr_title}")

        # Trigger learning for opened/synchronized PRs
        if action in ["opened", "synchronize"] and self.config.enable_auto_learn:
            if self.on_learn_trigger:
                learning_data = {
                    "source": "github_pr",
                    "action": action,
                    "pr_number": pr_number,
                    "title": pr_title,
                    "body": pr.get("body", ""),
                    "head_ref": pr.get("head", {}).get("ref", ""),
                    "base_ref": pr.get("base", {}).get("ref", ""),
                }
                try:
                    await self.on_learn_trigger(learning_data)
                    return {
                        "action": f"pr_{action}_processed",
                        "learning_triggered": True,
                        "pr_number": pr_number,
                    }
                except Exception as e:
                    logger.error(f"Learning trigger failed: {e}")

        return {
            "action": f"pr_{action}_logged",
            "pr_number": pr_number,
        }

    async def _handle_issues(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle issue events."""
        action = event.action
        issue = event.payload.get("issue", {})
        issue_number = issue.get("number")
        issue_title = issue.get("title", "")

        logger.info(f"Issue #{issue_number} {action}: {issue_title}")

        return {
            "action": f"issue_{action}_logged",
            "issue_number": issue_number,
        }

    async def _handle_issue_comment(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle issue comment events."""
        action = event.action
        comment = event.payload.get("comment", {})
        issue = event.payload.get("issue", {})
        issue_number = issue.get("number")

        logger.info(f"Comment on issue #{issue_number} by {event.sender}")

        # Check for commands in comments
        comment_body = comment.get("body", "")
        if comment_body.startswith("/flashai"):
            # Parse command
            parts = comment_body.split()
            command = parts[1] if len(parts) > 1 else "help"
            args = parts[2:] if len(parts) > 2 else []

            return {
                "action": "command_detected",
                "command": command,
                "args": args,
                "issue_number": issue_number,
            }

        return {
            "action": f"comment_{action}_logged",
            "issue_number": issue_number,
        }


def create_webhook_endpoint_code() -> str:
    """
    Generate code snippet for setting up webhook endpoint.

    This is a helper to show users how to integrate webhooks
    with their web framework.
    """
    return '''
# FastAPI webhook endpoint example
from fastapi import FastAPI, Request, Header, HTTPException

app = FastAPI()
webhook_handler = GitHubWebhookHandler(config)

@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(...),
    x_github_delivery: str = Header(...),
    x_hub_signature_256: str = Header(None),
):
    """Handle incoming GitHub webhooks."""
    raw_payload = await request.body()
    payload = await request.json()

    result = await webhook_handler.handle_webhook(
        event_type=x_github_event,
        payload=payload,
        delivery_id=x_github_delivery,
        signature=x_hub_signature_256,
        raw_payload=raw_payload,
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)

    return {"status": "ok", "result": result}
'''

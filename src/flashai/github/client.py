"""
GitHub API Client for FlashAI integration.

Provides methods to interact with GitHub repositories,
issues, pull requests, and other resources.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from dataclasses import dataclass

import httpx

from flashai.core.config import GitHubConfig


logger = logging.getLogger(__name__)


@dataclass
class GitHubFile:
    """Represents a file in a GitHub repository."""
    path: str
    content: str
    sha: str
    encoding: str = "utf-8"


@dataclass
class GitHubIssue:
    """Represents a GitHub issue."""
    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    author: str
    created_at: str
    updated_at: str


@dataclass
class GitHubPullRequest:
    """Represents a GitHub pull request."""
    number: int
    title: str
    body: str
    state: str
    head_ref: str
    base_ref: str
    author: str
    created_at: str
    updated_at: str
    mergeable: Optional[bool] = None


class GitHubClient:
    """
    Async GitHub API client for FlashAI.

    Handles authentication and provides methods for:
    - Repository operations
    - Issue management
    - Pull request operations
    - File content access
    """

    def __init__(
        self,
        config: GitHubConfig,
        token: Optional[str] = None,
    ):
        self.config = config
        self.token = token
        self.base_url = config.api_base_url

        # HTTP client configuration
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.config.api_timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _repo_url(self, path: str = "") -> str:
        """Build repository URL."""
        base = f"/repos/{self.config.repo_owner}/{self.config.repo_name}"
        return f"{base}/{path}" if path else base

    async def get_repo_info(self) -> dict[str, Any]:
        """Get repository information."""
        client = await self._get_client()
        response = await client.get(self._repo_url())
        response.raise_for_status()
        return response.json()

    async def get_file_content(
        self,
        path: str,
        ref: Optional[str] = None,
    ) -> GitHubFile:
        """
        Get file content from repository.

        Args:
            path: File path in repository
            ref: Git reference (branch, tag, commit)

        Returns:
            GitHubFile with content
        """
        client = await self._get_client()
        params = {}
        if ref:
            params["ref"] = ref

        response = await client.get(
            self._repo_url(f"contents/{path}"),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        # Decode content
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8")

        return GitHubFile(
            path=data["path"],
            content=content,
            sha=data["sha"],
            encoding=data.get("encoding", "utf-8"),
        )

    async def list_issues(
        self,
        state: str = "open",
        labels: Optional[list[str]] = None,
        limit: int = 30,
    ) -> list[GitHubIssue]:
        """List repository issues."""
        client = await self._get_client()
        params = {
            "state": state,
            "per_page": limit,
        }
        if labels:
            params["labels"] = ",".join(labels)

        response = await client.get(
            self._repo_url("issues"),
            params=params,
        )
        response.raise_for_status()

        issues = []
        for data in response.json():
            issues.append(GitHubIssue(
                number=data["number"],
                title=data["title"],
                body=data.get("body", ""),
                state=data["state"],
                labels=[l["name"] for l in data.get("labels", [])],
                author=data["user"]["login"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
            ))
        return issues

    async def get_issue(self, issue_number: int) -> GitHubIssue:
        """Get a specific issue."""
        client = await self._get_client()
        response = await client.get(
            self._repo_url(f"issues/{issue_number}")
        )
        response.raise_for_status()
        data = response.json()

        return GitHubIssue(
            number=data["number"],
            title=data["title"],
            body=data.get("body", ""),
            state=data["state"],
            labels=[l["name"] for l in data.get("labels", [])],
            author=data["user"]["login"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    async def create_issue_comment(
        self,
        issue_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Create a comment on an issue."""
        client = await self._get_client()
        response = await client.post(
            self._repo_url(f"issues/{issue_number}/comments"),
            json={"body": body},
        )
        response.raise_for_status()
        return response.json()

    async def list_pull_requests(
        self,
        state: str = "open",
        limit: int = 30,
    ) -> list[GitHubPullRequest]:
        """List repository pull requests."""
        client = await self._get_client()
        params = {
            "state": state,
            "per_page": limit,
        }

        response = await client.get(
            self._repo_url("pulls"),
            params=params,
        )
        response.raise_for_status()

        prs = []
        for data in response.json():
            prs.append(GitHubPullRequest(
                number=data["number"],
                title=data["title"],
                body=data.get("body", ""),
                state=data["state"],
                head_ref=data["head"]["ref"],
                base_ref=data["base"]["ref"],
                author=data["user"]["login"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                mergeable=data.get("mergeable"),
            ))
        return prs

    async def get_pull_request(self, pr_number: int) -> GitHubPullRequest:
        """Get a specific pull request."""
        client = await self._get_client()
        response = await client.get(
            self._repo_url(f"pulls/{pr_number}")
        )
        response.raise_for_status()
        data = response.json()

        return GitHubPullRequest(
            number=data["number"],
            title=data["title"],
            body=data.get("body", ""),
            state=data["state"],
            head_ref=data["head"]["ref"],
            base_ref=data["base"]["ref"],
            author=data["user"]["login"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            mergeable=data.get("mergeable"),
        )

    async def get_pull_request_files(
        self,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Get files changed in a pull request."""
        client = await self._get_client()
        response = await client.get(
            self._repo_url(f"pulls/{pr_number}/files")
        )
        response.raise_for_status()
        return response.json()

    async def create_pull_request_comment(
        self,
        pr_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Create a comment on a pull request."""
        # PR comments use the issues API
        return await self.create_issue_comment(pr_number, body)

    async def get_commits(
        self,
        sha: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get repository commits."""
        client = await self._get_client()
        params = {"per_page": limit}
        if sha:
            params["sha"] = sha

        response = await client.get(
            self._repo_url("commits"),
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def get_commit(self, sha: str) -> dict[str, Any]:
        """Get a specific commit."""
        client = await self._get_client()
        response = await client.get(
            self._repo_url(f"commits/{sha}")
        )
        response.raise_for_status()
        return response.json()

    async def get_branches(self) -> list[dict[str, Any]]:
        """Get repository branches."""
        client = await self._get_client()
        response = await client.get(self._repo_url("branches"))
        response.raise_for_status()
        return response.json()

    async def dispatch_workflow(
        self,
        workflow_id: str,
        ref: str,
        inputs: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Trigger a GitHub Actions workflow."""
        client = await self._get_client()
        payload = {"ref": ref}
        if inputs:
            payload["inputs"] = inputs

        response = await client.post(
            self._repo_url(f"actions/workflows/{workflow_id}/dispatches"),
            json=payload,
        )
        return response.status_code == 204

    async def create_repository_dispatch(
        self,
        event_type: str,
        client_payload: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create a repository dispatch event."""
        client = await self._get_client()
        payload = {"event_type": event_type}
        if client_payload:
            payload["client_payload"] = client_payload

        response = await client.post(
            self._repo_url("dispatches"),
            json=payload,
        )
        return response.status_code == 204

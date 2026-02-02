"""
FlashAI Command Line Interface

Provides CLI commands for:
- Starting the server
- Running reasoning queries
- Managing state (save/reset/restore)
- Configuring the system
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax

from flashai.utils.portable import (
    is_portable_mode,
    get_flash_drive_path,
    setup_portable_environment,
)


console = Console()


def get_base_path() -> Path:
    """Get base path based on portable mode."""
    if is_portable_mode():
        flash_drive = get_flash_drive_path()
        if flash_drive:
            setup_portable_environment()
            return flash_drive
    return Path.cwd()


@click.group()
@click.version_option(version="1.0.0", prog_name="FlashAI")
@click.option("--base-path", type=click.Path(exists=True), help="Base path override")
@click.pass_context
def main(ctx: click.Context, base_path: Optional[str]) -> None:
    """
    FlashAI - Portable AI System with Energy-Based Reasoning

    A self-contained AI that learns and adapts to users.
    """
    ctx.ensure_object(dict)
    ctx.obj["base_path"] = Path(base_path) if base_path else get_base_path()

    # Show portable mode banner
    if is_portable_mode():
        console.print(
            Panel.fit(
                "[bold green]Portable Mode Active[/bold green]\n"
                f"Flash Drive: {ctx.obj['base_path']}",
                title="FlashAI",
            )
        )


@main.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8420, help="Server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Start the FlashAI API server."""
    from flashai.server import run_server

    base_path = ctx.obj["base_path"]

    console.print(f"[bold]Starting FlashAI server...[/bold]")
    console.print(f"Base path: {base_path}")
    console.print(f"Server: http://{host}:{port}")
    console.print(f"API docs: http://{host}:{port}/docs")
    console.print()

    run_server(
        host=host,
        port=port,
        base_path=base_path,
        reload=reload,
    )


@main.command()
@click.argument("query")
@click.option("--context", "-c", type=str, help="JSON context")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def reason(
    ctx: click.Context,
    query: str,
    context: Optional[str],
    json_output: bool,
) -> None:
    """Perform reasoning on a query."""
    from flashai.core.engine import FlashAIEngine

    base_path = ctx.obj["base_path"]

    async def run_reasoning():
        engine = FlashAIEngine(base_path=base_path)
        await engine.initialize()

        ctx_dict = json.loads(context) if context else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Reasoning...", total=None)
            result = await engine.reason(query=query, context=ctx_dict)

        await engine.shutdown()
        return result

    result = asyncio.run(run_reasoning())

    if json_output:
        console.print(json.dumps(result, indent=2))
    else:
        console.print(Panel(f"[bold]Query:[/bold] {query}", title="Input"))

        # Show reasoning chain
        chain = result.get("chain", {})
        steps = chain.get("steps", [])

        if steps:
            table = Table(title="Reasoning Chain")
            table.add_column("Step", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Energy", style="yellow")
            table.add_column("Confidence", style="green")

            for step in steps:
                table.add_row(
                    str(step["step_id"]),
                    step["step_type"],
                    f"{step['energy']:.4f}",
                    f"{step['confidence']:.2%}",
                )

            console.print(table)

        # Show final result
        console.print(Panel(
            f"[bold]Energy:[/bold] {result.get('final_energy', 'N/A'):.4f}\n"
            f"[bold]Confidence:[/bold] {result.get('confidence', 'N/A'):.2%}",
            title="Result",
        ))


@main.command()
@click.option("--name", "-n", help="Export name")
@click.option("--no-user-data", is_flag=True, help="Exclude user data")
@click.pass_context
def reset(ctx: click.Context, name: Optional[str], no_user_data: bool) -> None:
    """Save current state and reset to initial state."""
    from flashai.core.engine import FlashAIEngine

    base_path = ctx.obj["base_path"]

    async def run_reset():
        engine = FlashAIEngine(base_path=base_path)
        result = await engine.save_and_reset(
            export_name=name,
            include_user_data=not no_user_data,
        )
        return result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Saving and resetting...", total=None)
        result = asyncio.run(run_reset())

    console.print(Panel(
        f"[bold green]Reset complete![/bold green]\n\n"
        f"Export saved to: {result['export']['path']}\n"
        f"Files exported: {result['export']['files_count']}",
        title="FlashAI Reset",
    ))


@main.command()
@click.argument("export_path", type=click.Path(exists=True))
@click.pass_context
def restore(ctx: click.Context, export_path: str) -> None:
    """Restore from an export file."""
    from flashai.core.engine import FlashAIEngine

    base_path = ctx.obj["base_path"]

    async def run_restore():
        engine = FlashAIEngine(base_path=base_path)
        result = await engine.restore_from_export(export_path)
        return result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Restoring...", total=None)
        result = asyncio.run(run_restore())

    console.print(Panel(
        f"[bold green]Restore complete![/bold green]\n\n"
        f"Files restored: {result['files_restored']}",
        title="FlashAI Restore",
    ))


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    from flashai.core.engine import FlashAIEngine

    base_path = ctx.obj["base_path"]

    async def get_status():
        engine = FlashAIEngine(base_path=base_path)
        return await engine.get_status()

    status_info = asyncio.run(get_status())

    table = Table(title="FlashAI Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in status_info.items():
        table.add_row(key, str(value))

    console.print(table)


@main.command()
@click.pass_context
def exports(ctx: click.Context) -> None:
    """List available exports."""
    from flashai.state.manager import StateManager

    base_path = ctx.obj["base_path"]

    async def list_exports():
        manager = StateManager(
            base_path=base_path,
            checkpoints_path=base_path / "data" / "checkpoints",
            exports_path=base_path / "data" / "exports",
        )
        return await manager.list_exports()

    export_list = asyncio.run(list_exports())

    if not export_list:
        console.print("[yellow]No exports found.[/yellow]")
        return

    table = Table(title="Available Exports")
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Size", style="yellow")

    for export in export_list:
        manifest = export.get("manifest", {})
        size_mb = export.get("size_bytes", 0) / (1024 * 1024)
        table.add_row(
            manifest.get("export_name", "Unknown"),
            manifest.get("created_at", "Unknown")[:19],
            f"{size_mb:.2f} MB",
        )

    console.print(table)


@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize FlashAI in the current directory."""
    from flashai.core.config import FlashAIConfig
    from flashai.utils.portable import create_portable_marker

    base_path = ctx.obj["base_path"]

    # Create directory structure
    directories = [
        "config",
        "data/user_profiles",
        "data/checkpoints",
        "data/exports",
        "data/logs",
        "models",
    ]

    for dir_name in directories:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)

    # Create default config
    config = FlashAIConfig()
    config_path = base_path / "config" / "flashai.yaml"
    config.save(config_path)

    # Create portable marker
    create_portable_marker(base_path)

    console.print(Panel(
        f"[bold green]FlashAI initialized![/bold green]\n\n"
        f"Location: {base_path}\n"
        f"Config: {config_path}\n\n"
        f"Run [bold]flashai serve[/bold] to start the server.",
        title="Initialization Complete",
    ))


@main.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show current configuration."""
    from flashai.core.config import FlashAIConfig

    base_path = ctx.obj["base_path"]
    config_path = base_path / "config" / "flashai.yaml"

    if config_path.exists():
        cfg = FlashAIConfig.load(config_path)
    else:
        cfg = FlashAIConfig()

    config_json = json.dumps(cfg.model_dump(), indent=2)
    syntax = Syntax(config_json, "json", theme="monokai", line_numbers=True)

    console.print(Panel(syntax, title="FlashAI Configuration"))


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--preview", is_flag=True, help="Preview without applying")
@click.option("--format", "file_format", type=str, help="Force file format (json, csv, yaml, txt, md)")
@click.pass_context
def learn(ctx: click.Context, file_path: str, preview: bool, file_format: Optional[str]) -> None:
    """Learn from a data file."""
    from flashai.core.engine import FlashAIEngine
    from flashai.learning.file_parser import LearningDataParser, FileFormat

    base_path = ctx.obj["base_path"]
    file_path = Path(file_path)

    # Parse the file
    parser = LearningDataParser()

    format_hint = None
    if file_format:
        format_map = {
            "json": FileFormat.JSON,
            "jsonl": FileFormat.JSONL,
            "csv": FileFormat.CSV,
            "yaml": FileFormat.YAML,
            "txt": FileFormat.TXT,
            "md": FileFormat.MARKDOWN,
        }
        format_hint = format_map.get(file_format.lower())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Parsing file...", total=None)

        if format_hint:
            content = file_path.read_text()
            result = parser.parse_content(content, filename=file_path.name, format_hint=format_hint)
        else:
            result = parser.parse_file(file_path)

    if not result.success:
        console.print(Panel(
            f"[bold red]Failed to parse file[/bold red]\n\n"
            f"Errors: {', '.join(result.errors)}",
            title="Parse Error",
        ))
        return

    # Show preview
    console.print(Panel(
        f"[bold]File:[/bold] {file_path.name}\n"
        f"[bold]Format:[/bold] {result.format_detected.value}\n"
        f"[bold]Examples found:[/bold] {len(result.examples)}\n"
        f"[bold]Warnings:[/bold] {len(result.warnings)}",
        title="Parse Result",
    ))

    if result.examples:
        table = Table(title="Examples Preview (first 5)")
        table.add_column("Query", style="cyan", max_width=40)
        table.add_column("Answer", style="green", max_width=40)

        for ex in result.examples[:5]:
            table.add_row(
                ex.query[:40] + "..." if len(ex.query) > 40 else ex.query,
                ex.answer[:40] + "..." if len(ex.answer) > 40 else ex.answer,
            )

        console.print(table)

    if preview:
        console.print("[yellow]Preview mode - no changes applied[/yellow]")
        return

    # Apply learning
    async def apply_learning():
        engine = FlashAIEngine(base_path=base_path)
        await engine.initialize()

        learning_data = result.to_learning_data()
        learn_result = await engine.learn(data=learning_data)

        await engine.shutdown()
        return learn_result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Learning from data...", total=None)
        learn_result = asyncio.run(apply_learning())

    console.print(Panel(
        f"[bold green]Learning complete![/bold green]\n\n"
        f"Examples processed: {learn_result.get('examples_processed', 0)}\n"
        f"Average loss: {learn_result.get('average_loss', 0):.4f}",
        title="Learning Result",
    ))


@main.command()
@click.pass_context
def formats(ctx: click.Context) -> None:
    """Show supported file formats for learning."""
    from flashai.learning.file_parser import LearningDataParser

    parser = LearningDataParser()
    supported = parser.get_supported_formats()

    console.print(Panel.fit("[bold]Supported Learning Data Formats[/bold]"))

    for fmt in supported:
        console.print(f"\n[bold cyan]{fmt['format'].upper()}[/bold cyan] ({', '.join(fmt['extensions'])})")
        console.print(f"  {fmt['description']}")
        console.print(f"  [dim]Example: {fmt['example'][:60]}...[/dim]")


@main.command()
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def webhook_setup(ctx: click.Context, output: Optional[str]) -> None:
    """Generate webhook setup instructions."""
    from flashai.github.webhooks import create_webhook_endpoint_code

    base_path = ctx.obj["base_path"]

    instructions = f"""
# GitHub Webhook Setup for FlashAI

## 1. Configure FlashAI

Edit your config file at: {base_path / 'config' / 'flashai.yaml'}

```yaml
github:
  repo_owner: YOUR_USERNAME
  repo_name: YOUR_REPO
  webhook_secret: YOUR_SECRET_HERE
  webhook_events:
    - push
    - pull_request
    - issues
    - issue_comment
```

## 2. Start the FlashAI Server

```bash
flashai serve --host 0.0.0.0 --port 8420
```

## 3. Configure GitHub Webhook

1. Go to your repository settings
2. Navigate to Webhooks
3. Click "Add webhook"
4. Configure:
   - Payload URL: http://YOUR_SERVER:8420/webhooks/github
   - Content type: application/json
   - Secret: YOUR_SECRET_HERE
   - Events: Select individual events (push, pull requests, issues)

## 4. Webhook Endpoint Code

{create_webhook_endpoint_code()}

## 5. Environment Variables

For production, set these environment variables:

```bash
export FLASHAI_GITHUB_TOKEN=your_github_token
export FLASHAI_WEBHOOK_SECRET=your_webhook_secret
export FLASHAI_REPO_OWNER=your_username
export FLASHAI_REPO_NAME=your_repo
```
"""

    if output:
        Path(output).write_text(instructions)
        console.print(f"[green]Instructions saved to {output}[/green]")
    else:
        console.print(instructions)


if __name__ == "__main__":
    main()

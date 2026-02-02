# FlashAI Training Guide

Complete guide to training and customizing your personal FlashAI assistant.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding Energy-Based Reasoning](#understanding-energy-based-reasoning)
3. [Training Your AI](#training-your-ai)
4. [Data Formats](#data-formats)
5. [Personalization Options](#personalization-options)
6. [Advanced Features](#advanced-features)
7. [Agentic Capabilities](#agentic-capabilities)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Initial Setup

1. **Copy FlashAI to your flash drive:**
   ```bash
   # From the repository
   cp -r FlashAI /path/to/your/flashdrive/
   ```

2. **Initialize the system:**
   ```bash
   cd /path/to/flashdrive/FlashAI
   ./scripts/launch.sh init   # Linux/Mac
   # or
   scripts\launch.bat init    # Windows
   ```

3. **Start the server:**
   ```bash
   flashai serve --port 8420
   ```

4. **Open the web interface:**
   Navigate to `http://localhost:8420` in your browser.

### First-Time User Registration

When you first plug in your flash drive and start FlashAI:

1. The system automatically detects you as a new user
2. A unique profile is created based on your machine identifier
3. Your preferences and learning history will be stored encrypted
4. The AI begins adapting to your interaction patterns

---

## Understanding Energy-Based Reasoning

FlashAI uses **Energy-Based Models (EBMs)** for reasoning, inspired by [Logical Intelligence's Kona](https://logicalintelligence.com/kona-ebms-energy-based-models).

### How It Works

Unlike traditional AI that predicts the "most likely" answer, FlashAI:

1. **Maps solutions to energy values** - Lower energy = better solution
2. **Minimizes energy through optimization** - Finds the best answer by gradient descent
3. **Enforces constraints** - Ensures answers satisfy logical requirements
4. **Learns from mistakes** - Adjusts energy landscape based on feedback

### Key Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Energy** | Solution quality score (lower is better) | < 0.5 |
| **Confidence** | Certainty in the answer | > 70% |
| **Coherence** | Logical consistency | > 0.8 |

### Viewing Energy Breakdown

```bash
# CLI
flashai reason "Your question here" --json-output

# API
GET /api/v1/model/energy-breakdown?query=...&answer=...
```

---

## Training Your AI

### Method 1: Interactive Learning

The easiest way - just use the AI and provide feedback:

1. Ask questions through the chat interface
2. The AI learns from your interaction patterns
3. Provide explicit feedback when answers are wrong
4. Your preferences are automatically saved

### Method 2: File Upload

Upload structured training data:

**Via Web Interface:**
1. Go to the "Learn" tab
2. Drag and drop your file
3. Preview the parsed examples
4. Click "Apply" to train

**Via CLI:**
```bash
# Preview data
flashai learn training_data.json --preview

# Apply learning
flashai learn training_data.json
```

**Via API:**
```bash
curl -X POST http://localhost:8420/api/v1/learn/upload \
  -F "file=@training_data.json" \
  -F "apply_immediately=true"
```

### Method 3: Programmatic Training

```python
from flashai.core.engine import FlashAIEngine

engine = FlashAIEngine()
await engine.initialize()

# Train with examples
await engine.learn({
    "examples": [
        {
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of AI...",
            "negative_samples": ["ML is unrelated to computers"]
        }
    ]
})
```

---

## Data Formats

FlashAI supports multiple training data formats:

### JSON

```json
[
  {
    "query": "What is the capital of France?",
    "answer": "Paris is the capital of France.",
    "context": {"category": "geography"},
    "negative_samples": ["London", "Berlin"]
  }
]
```

### JSONL (JSON Lines)

```jsonl
{"query": "Q1", "answer": "A1"}
{"query": "Q2", "answer": "A2"}
```

### CSV

```csv
query,answer,context
"What is 2+2?","4","math"
"What color is the sky?","Blue","general"
```

### YAML

```yaml
examples:
  - query: What is Python?
    answer: Python is a programming language.
  - query: What is JavaScript?
    answer: JavaScript is a web programming language.
```

### Markdown

```markdown
# What is Machine Learning?

Machine learning is a type of artificial intelligence...

# How do neural networks work?

Neural networks are computing systems inspired by biological...
```

### Plain Text

```
Q: What is AI?
A: AI stands for Artificial Intelligence...

Q: What is deep learning?
A: Deep learning uses multi-layer neural networks...
```

---

## Personalization Options

### Settings Overview

Access settings via:
- Web UI: Settings tab
- CLI: `flashai config`
- API: `GET/PUT /api/v1/settings`

### Available Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `dark_mode` | Use dark theme | true |
| `compact_view` | Reduce UI spacing | false |
| `auto_learn` | Learn from interactions | true |
| `screen_learning` | Learn from screen captures | false |
| `reasoning_depth` | Max reasoning steps | 10 |
| `confidence_threshold` | Min confidence for answers | 0.7 |
| `save_history` | Store interaction history | true |
| `encrypt_data` | Encrypt stored data | true |

### Customizing Behavior

**Adjust Reasoning Depth:**
```python
# More thorough reasoning (slower)
await engine.reason(query, constraints=["be_thorough"])

# Quick reasoning (faster)
await engine.reason(query, constraints=["be_brief"])
```

**Set Domain Focus:**
```yaml
# config/flashai.yaml
ebm:
  domain_weights:
    technical: 1.5
    creative: 0.8
    analytical: 1.2
```

---

## Advanced Features

### Notes & Memory

Store important information for the AI to remember:

```python
# Create a note
await engine.user_manager.create_note(
    user_id=user_id,
    content="My project uses Python 3.11",
    title="Tech Stack",
    tags=["project", "python"]
)

# Search notes
notes = await engine.user_manager.search_notes(
    user_id=user_id,
    query="python"
)
```

### Screen Learning

Enable the AI to learn from screen captures (requires user consent):

1. Enable in settings: `screen_learning: true`
2. Capture screen via API:
   ```bash
   curl -X POST http://localhost:8420/api/v1/learn/screen \
     -F "image=@screenshot.png" \
     -F "context={\"app\": \"vscode\"}"
   ```

### Save and Reset

Export your trained AI and reset to initial state:

```bash
# Save current state and reset
flashai reset --name my_backup

# Restore from backup
flashai restore data/exports/my_backup_20240101_120000.zip
```

### GitHub Integration

Connect to GitHub for automatic learning from repositories:

1. Configure webhook in `config/flashai.yaml`:
   ```yaml
   github:
     repo_owner: your_username
     repo_name: your_repo
     webhook_secret: your_secret
     webhook_events:
       - push
       - pull_request
   ```

2. Set up webhook in GitHub repository settings
3. FlashAI will learn from code changes automatically

---

## Agentic Capabilities

FlashAI includes a framework for building autonomous AI agents.

### Creating an Agent

```python
from flashai.agents import Agent, AgentConfig, AgentCapability
from flashai.agents.orchestrator import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator(engine=engine)

# Define agent configuration
config = AgentConfig(
    name="research_agent",
    description="Agent for research tasks",
    capabilities=[
        AgentCapability.REASONING,
        AgentCapability.TOOL_USE,
        AgentCapability.WEB_SEARCH,
    ],
    max_iterations=20,
    timeout_seconds=300,
)

# Create agent
agent = orchestrator.create_agent(config)

# Run a task
result = await agent.run("Research the latest trends in AI")
```

### Available Capabilities

| Capability | Description |
|------------|-------------|
| `REASONING` | Use energy-based reasoning |
| `LEARNING` | Learn from interactions |
| `TOOL_USE` | Use external tools |
| `WEB_SEARCH` | Search the web |
| `FILE_ACCESS` | Read/write files |
| `CODE_EXECUTION` | Execute Python code |
| `MEMORY` | Persistent memory |
| `COMMUNICATION` | Inter-agent messaging |
| `SCREEN_ACCESS` | Capture/analyze screen |

### Built-in Tools

| Tool | Description |
|------|-------------|
| `file_read` | Read file contents |
| `file_write` | Write to files |
| `web_search` | Search the web |
| `execute_code` | Run Python code (sandboxed) |
| `calculator` | Mathematical calculations |

### Creating Custom Tools

```python
from flashai.agents.tools import Tool, ToolDefinition, ToolParameter

class MyCustomTool(Tool):
    def __init__(self):
        definition = ToolDefinition(
            name="my_tool",
            description="Does something useful",
            parameters=[
                ToolParameter(
                    name="input",
                    description="Input value",
                    type="string",
                ),
            ],
        )
        super().__init__(definition)

    async def execute(self, input: str) -> str:
        # Your tool logic here
        return f"Processed: {input}"

# Register with orchestrator
orchestrator.tool_registry.register(MyCustomTool())
```

### Multi-Agent Workflows

```python
# Run tasks in parallel
results = await orchestrator.run_parallel([
    ("Research topic A", None),
    ("Research topic B", None),
])

# Run tasks sequentially
results = await orchestrator.run_sequential([
    "Gather requirements",
    "Design solution",
    "Implement code",
])
```

---

## Troubleshooting

### Common Issues

**AI not learning from interactions:**
- Check `auto_learn` setting is enabled
- Verify user profile is created (`flashai status`)
- Check logs in `data/logs/flashai.log`

**High energy scores (> 1.0):**
- Provide more training examples
- Reduce reasoning depth temporarily
- Check constraint definitions

**File upload fails:**
- Verify file format is supported
- Check file encoding (UTF-8 recommended)
- Preview file first with `--preview` flag

**Slow responses:**
- Reduce `reasoning_depth` setting
- Use `compact` environment config
- Check available system resources

### Getting Help

- Check logs: `tail -f data/logs/flashai.log`
- View status: `flashai status`
- Test API: `curl http://localhost:8420/health`

### Reset If Needed

```bash
# Complete reset (saves backup first)
flashai reset --name emergency_backup

# Restore known good state
flashai restore path/to/backup.zip
```

---

## Quick Reference

### CLI Commands

| Command | Description |
|---------|-------------|
| `flashai init` | Initialize FlashAI |
| `flashai serve` | Start the server |
| `flashai reason "query"` | Run a query |
| `flashai learn file.json` | Learn from file |
| `flashai formats` | Show supported formats |
| `flashai status` | Show system status |
| `flashai reset` | Save and reset |
| `flashai restore file.zip` | Restore from export |
| `flashai exports` | List available exports |
| `flashai config` | Show configuration |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/initialize` | POST | Initialize session |
| `/api/v1/reason` | POST | Perform reasoning |
| `/api/v1/learn` | POST | Learn from examples |
| `/api/v1/learn/upload` | POST | Upload training file |
| `/api/v1/save-and-reset` | POST | Save and reset |
| `/api/v1/restore` | POST | Restore from export |
| `/api/v1/settings` | GET/PUT | User settings |
| `/api/v1/notes` | GET/POST | User notes |
| `/api/v1/history` | GET | Interaction history |

---

*FlashAI - Your Personal, Portable AI Assistant*

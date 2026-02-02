# FlashAI

**Portable AI System with Energy-Based Reasoning**

FlashAI is a self-contained AI system designed to run from a flash drive, featuring energy-based reasoning inspired by [Kona EBMs](https://logicalintelligence.com/kona-ebms-energy-based-models). It learns and adapts to the first user who plugs in the flash drive, with built-in save/reset functionality and GitHub integration.

## Features

- **Energy-Based Reasoning**: Uses energy minimization for logical reasoning, finding solutions that satisfy constraints rather than just predicting likely outputs
- **User Adaptation**: Learns user preferences and patterns from interactions
- **Portable by Design**: Runs entirely from a flash drive with no installation required
- **State Management**: Save current state to zip, reset to initial state, or restore from backup
- **GitHub Integration**: Webhooks for auto-learning from repository changes
- **REST API**: Full HTTP API for integration with other applications
- **Web Interface**: Modern, intuitive UI with history, notes, and settings
- **File Learning**: Upload training data in JSON, CSV, YAML, Markdown, or text formats
- **Notes & Memory**: Save important information for the AI to remember
- **Agentic Framework**: Build autonomous agents with tools, memory, and multi-agent coordination
- **Screen Learning**: Optional screen capture learning (requires user consent)
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Documentation

- **[Training Guide](docs/TRAINING_GUIDE.md)**: Complete guide to training and customizing your AI
- **[API Reference](#api-reference)**: REST API endpoints and usage

## Quick Start

### Option 1: Portable Installation (Flash Drive)

1. Download the latest portable release from [Releases](https://github.com/moore70344/FlashAI/releases)
2. Extract to your flash drive
3. Run the launcher:

```bash
# Linux/macOS
./scripts/launch.sh

# Windows
scripts\launch.bat
```

### Option 2: Python Package Installation

```bash
pip install flashai
```

Or install from source:

```bash
git clone https://github.com/moore70344/FlashAI.git
cd FlashAI
pip install -e .
```

### Initialize and Start

```bash
# Initialize FlashAI
flashai init

# Start the server
flashai serve

# Or with custom settings
flashai serve --host 0.0.0.0 --port 8080
```

## Usage

### Command Line Interface

```bash
# Perform reasoning
flashai reason "What is the best approach to solve X problem?"

# Check status
flashai status

# Save state and reset
flashai reset --name my_backup

# Restore from backup
flashai restore data/exports/my_backup_20240101_120000.zip

# List exports
flashai exports

# View configuration
flashai config
```

### REST API

Start the server and access the API:

```bash
flashai serve
```

#### Initialize Session

```bash
curl -X POST http://localhost:8420/api/v1/initialize \
  -H "Content-Type: application/json" \
  -d '{"user_id": "optional-user-id"}'
```

#### Perform Reasoning

```bash
curl -X POST http://localhost:8420/api/v1/reason \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How should I structure my application?",
    "context": {"domain": "web development"},
    "constraints": ["must be scalable", "use microservices"]
  }'
```

#### Learn from Examples

```bash
curl -X POST http://localhost:8420/api/v1/learn \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {
        "query": "What is 2+2?",
        "answer": "4",
        "negative_samples": ["3", "5", "22"]
      }
    ]
  }'
```

#### Save and Reset

```bash
curl -X POST http://localhost:8420/api/v1/save-and-reset \
  -H "Content-Type: application/json" \
  -d '{"export_name": "my_backup", "include_user_data": true}'
```

### Python API

```python
import asyncio
from flashai import FlashAIEngine

async def main():
    # Initialize engine
    engine = FlashAIEngine()
    await engine.initialize()

    # Perform reasoning
    result = await engine.reason(
        query="What is the optimal solution?",
        context={"domain": "optimization"},
        constraints=["minimize cost", "maximize efficiency"],
    )

    print(f"Energy: {result['final_energy']}")
    print(f"Confidence: {result['confidence']}")

    # Learn from feedback
    await engine.learn({
        "examples": [
            {"query": "example", "answer": "correct", "negative_samples": ["wrong"]}
        ]
    })

    # Save and reset
    export = await engine.save_and_reset(export_name="backup")
    print(f"Exported to: {export['export']['path']}")

    await engine.shutdown()

asyncio.run(main())
```

## GitHub Integration

### Setting Up Webhooks

1. Configure FlashAI with your repository details:

```yaml
# config/flashai.yaml
github:
  repo_owner: your-username
  repo_name: your-repo
  webhook_secret: your-secret-here
  webhook_events:
    - push
    - pull_request
    - issues
```

2. Start the server with external access:

```bash
flashai serve --host 0.0.0.0 --port 8420
```

3. Add webhook in GitHub repository settings:
   - Payload URL: `http://your-server:8420/webhooks/github`
   - Content type: `application/json`
   - Secret: `your-secret-here`

4. FlashAI will automatically learn from:
   - Code pushes
   - Pull request changes
   - Issue discussions

### Environment Variables

```bash
export FLASHAI_GITHUB_TOKEN=your_github_token
export FLASHAI_WEBHOOK_SECRET=your_webhook_secret
export FLASHAI_REPO_OWNER=your_username
export FLASHAI_REPO_NAME=your_repo
```

## Architecture

### Energy-Based Reasoning Model

FlashAI uses an Energy-Based Model (EBM) for reasoning, inspired by recent advances like [Logical Intelligence's Kona](https://www.businesswire.com/news/home/20260120751310/en/Logical-Intelligence-Introduces-First-Energy-Based-Reasoning-AI-Model-Signals-Early-Steps-Toward-AGI-Adds-Yann-LeCun-and-Patrick-Hillmann-to-Leadership).

Key concepts:
- **Energy Function**: Maps input-output pairs to scalar energy values
- **Low Energy = Good**: Correct/valid solutions have lower energy
- **Inference by Minimization**: Find answers by minimizing energy
- **Learning from Mistakes**: Model learns by recognizing and correcting errors

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│   Encode    │────▶│   Energy    │
│             │     │             │     │  Function   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   Answer    │◀────│  Minimize   │◀───────────┘
│             │     │   Energy    │
└─────────────┘     └─────────────┘
```

### Project Structure

```
FlashAI/
├── config/                    # Configuration files
│   ├── flashai.yaml          # Main configuration
│   ├── environments/         # Environment-specific configs
│   └── models_registry.yaml  # Available models
├── data/                     # Runtime data (auto-created)
│   ├── user_profiles/        # User adaptation data
│   ├── checkpoints/          # Model checkpoints
│   ├── exports/              # State exports
│   └── logs/                 # Application logs
├── models/                   # Trained model weights
├── scripts/                  # Launcher scripts
│   ├── launch.sh            # Unix launcher
│   └── launch.bat           # Windows launcher
├── src/flashai/             # Source code
│   ├── core/                # Core engine and config
│   ├── models/              # EBM implementation
│   ├── learning/            # User adaptation
│   ├── state/               # State management
│   ├── github/              # GitHub integration
│   ├── server/              # REST API
│   └── utils/               # Utilities
└── tests/                   # Test suite
```

## Configuration

### Main Configuration (`config/flashai.yaml`)

```yaml
# Energy-Based Model
ebm:
  input_dim: 512
  hidden_dims: [256, 128, 64]
  inference_steps: 10
  max_reasoning_depth: 5

# User Adaptation
user_adaptation:
  max_interaction_history: 1000
  adaptation_rate: 0.1
  encrypt_profile: true

# Server
server:
  host: "127.0.0.1"
  port: 8420

# Runtime
device: "auto"  # auto, cpu, cuda, mps
```

### Environment Configs

Use different configurations for different scenarios:

```bash
# Development (verbose, no encryption)
cp config/environments/development.yaml config/flashai.yaml

# Production (optimized, secure)
cp config/environments/production.yaml config/flashai.yaml

# Portable (lightweight, CPU-only)
cp config/environments/portable.yaml config/flashai.yaml
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/moore70344/FlashAI.git
cd FlashAI

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/
black --check src/
mypy src/flashai
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=flashai --cov-report=html

# Specific test file
pytest tests/test_ebm.py -v
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/initialize` | Initialize user session |
| GET | `/api/v1/status` | System status |
| POST | `/api/v1/reason` | Perform reasoning |
| POST | `/api/v1/learn` | Learn from examples |
| POST | `/api/v1/save-and-reset` | Export and reset |
| POST | `/api/v1/restore` | Restore from export |
| GET | `/api/v1/exports` | List exports |
| GET | `/api/v1/checkpoints` | List checkpoints |
| POST | `/webhooks/github` | GitHub webhook handler |

### Interactive Docs

When the server is running, visit:
- Swagger UI: http://localhost:8420/docs
- ReDoc: http://localhost:8420/redoc

## Agentic Framework

FlashAI includes a framework for building autonomous AI agents:

```python
from flashai.agents import AgentConfig, AgentCapability
from flashai.agents.orchestrator import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator(engine=engine)

# Create an agent
config = AgentConfig(
    name="research_agent",
    capabilities=[
        AgentCapability.REASONING,
        AgentCapability.TOOL_USE,
        AgentCapability.WEB_SEARCH,
    ],
)
agent = orchestrator.create_agent(config)

# Run a task
result = await agent.run("Research AI trends for 2024")
```

### Features

- **Multiple Agent Types**: Reasoning, tool-using, and custom agents
- **Built-in Tools**: File I/O, web search, code execution, calculator
- **Agent Memory**: Short-term and long-term memory systems
- **Multi-Agent Coordination**: Run tasks in parallel or sequence
- **Custom Tools**: Easy framework for adding new capabilities

See the [Training Guide](docs/TRAINING_GUIDE.md#agentic-capabilities) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Acknowledgments

- Inspired by [Logical Intelligence's Kona EBMs](https://logicalintelligence.com/kona-ebms-energy-based-models)
- Built with [PyTorch](https://pytorch.org/), [FastAPI](https://fastapi.tiangolo.com/), and [Click](https://click.palletsprojects.com/)

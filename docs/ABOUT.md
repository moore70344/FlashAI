# About FlashAI

## Vision

**FlashAI aims to be the first truly personal, portable AI assistant that you own and control.**

In a world where AI is increasingly centralized in cloud services controlled by large corporations, FlashAI offers a different path: an AI that lives on your flash drive, learns from you, adapts to your needs, and goes wherever you go—completely offline if you choose.

## What Makes FlashAI Different

### 1. Truly Portable
Unlike cloud-based AI assistants that require internet connectivity and send your data to remote servers, FlashAI runs entirely from a USB flash drive. Plug it into any compatible computer, and your personal AI is ready to help—no installation, no account creation, no data leaving your device.

### 2. You Own Your AI
FlashAI creates a unique profile for the first user who activates it. This AI becomes yours:
- It learns your preferences, communication style, and common tasks
- All data stays encrypted on your flash drive
- You can export, backup, or reset at any time
- No subscription fees or usage limits

### 3. Energy-Based Reasoning
FlashAI uses a fundamentally different approach to AI reasoning inspired by [Energy-Based Models](https://logicalintelligence.com/kona-ebms-energy-based-models):

| Traditional LLMs | FlashAI's EBM Approach |
|-----------------|------------------------|
| Predicts most likely next token | Finds solutions that minimize energy (error) |
| Can hallucinate confidently | Provides confidence scores based on energy |
| Difficult to constrain | Enforces explicit constraints |
| Black box reasoning | Transparent reasoning chains |

This means FlashAI aims for **correctness over probability**—it tries to find answers that are verifiably good rather than just statistically likely.

### 4. Grows With You
The more you use FlashAI, the better it understands you:
- Learns from every interaction
- Adapts to your domain expertise
- Remembers your notes and preferences
- Can be trained with your own documents, code, and data

### 5. Privacy First
- All processing happens locally on your device
- User profiles are encrypted with device-specific keys
- No telemetry, no tracking, no cloud dependencies
- You decide what the AI learns and remembers

## Project Goals

### Short-Term Goals
- [x] Core energy-based reasoning engine
- [x] User profile learning and adaptation
- [x] Save/reset/restore state management
- [x] File upload and processing for learning
- [x] Web interface for easy interaction
- [x] GitHub integration for developer workflows
- [x] Agentic framework for autonomous tasks

### Medium-Term Goals
- [ ] Improved natural language understanding
- [ ] Multi-modal learning (images, audio, video)
- [ ] Plugin system for extensions
- [ ] Mobile companion app
- [ ] Collaborative multi-user mode
- [ ] Fine-tuning with local compute

### Long-Term Vision
- [ ] Full offline language model integration
- [ ] Cross-device sync (encrypted, user-controlled)
- [ ] Specialized domain models (medical, legal, technical)
- [ ] Hardware acceleration support
- [ ] Community model marketplace
- [ ] Enterprise deployment options

## Use Cases

### Personal Assistant
- Answer questions based on your documents
- Remember important information
- Help with writing and research
- Learn your preferences over time

### Developer Tool
- Understand your codebase
- Generate code suggestions
- Review and explain code
- Integrate with GitHub workflows

### Learning Companion
- Study with AI-generated questions
- Get explanations tailored to your level
- Track your learning progress
- Build personalized knowledge bases

### Privacy-Sensitive Environments
- Work with confidential documents
- Healthcare and legal applications
- Air-gapped networks
- Compliance-restricted industries

## Technical Philosophy

### Why Energy-Based Models?
Traditional large language models (LLMs) work by predicting the most probable next token. While powerful, this approach has limitations:

1. **Hallucinations**: LLMs can generate confident-sounding but incorrect information
2. **Constraint Handling**: Difficult to ensure outputs satisfy specific requirements
3. **Reasoning Verification**: Hard to verify if the reasoning process was sound

Energy-Based Models offer a different paradigm:

```
Traditional LLM:  Input → Predict Most Likely Output
EBM Approach:     Input + Output → Compute Energy (Error Score)
                  Find Output that Minimizes Energy
```

By framing reasoning as optimization, FlashAI can:
- Provide meaningful confidence scores
- Enforce hard constraints on outputs
- Show reasoning steps with energy breakdowns
- Learn from mistakes more effectively

### Why Portable?
The flash drive form factor isn't just a gimmick—it embodies our philosophy:

1. **Ownership**: Your AI is a physical object you possess
2. **Privacy**: Data never leaves your control
3. **Accessibility**: Works on any compatible computer
4. **Resilience**: No dependency on internet or cloud services
5. **Personalization**: One device, one user, fully customized

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                         FlashAI                              │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Web UI    │  │    CLI      │  │  REST API   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   FlashAI Engine                       │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │  │    EBM      │ │   User      │ │   State     │     │  │
│  │  │  Reasoning  │ │  Profiles   │ │  Manager    │     │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │                Agentic Framework                       │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │  │
│  │  │ Agents  │ │  Tools  │ │ Memory  │ │Orchestr.│     │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │               File Processing                          │  │
│  │  Images │ Documents │ Code │ Data │ Archives │ Markup │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│         ┌────────────────┼────────────────┐                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Models    │  │   Config    │  │    Data     │          │
│  │  (weights)  │  │   (yaml)    │  │ (profiles)  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Flash Drive   │
                    │   (Portable)    │
                    └─────────────────┘
```

## Contributing

FlashAI is designed to be extensible and community-driven. We welcome contributions in:

- **Core Development**: Improving the reasoning engine, user adaptation, and performance
- **File Processors**: Adding support for more file formats
- **Tools**: Creating new tools for the agentic framework
- **Documentation**: Improving guides and examples
- **Testing**: Expanding test coverage
- **Integrations**: Building plugins and extensions

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Acknowledgments

FlashAI builds on ideas and research from:

- **[Logical Intelligence](https://logicalintelligence.com/)**: Pioneers of energy-based reasoning with Kona
- **[Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun)**: Long-time advocate for energy-based models
- **PyTorch Team**: For the deep learning framework
- **FastAPI**: For the excellent web framework
- **Open Source Community**: For countless libraries and tools

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/moore70344/FlashAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/moore70344/FlashAI/discussions)
- **Documentation**: [docs/](.)

---

## FAQ

### Is FlashAI a replacement for ChatGPT/Claude/etc.?

FlashAI is complementary rather than a replacement. While cloud-based LLMs excel at general knowledge and complex language tasks, FlashAI focuses on:
- Personal adaptation to your specific needs
- Privacy-preserving local processing
- Constraint-satisfying reasoning
- Portable, offline capability

### How much storage does it need?

A minimal FlashAI installation requires approximately:
- Base system: ~500 MB
- Default models: ~1-2 GB
- User data: Varies (typically < 100 MB)

A 4GB+ flash drive is recommended for comfortable operation.

### Can I use it without a flash drive?

Yes! While designed for portability, FlashAI works perfectly on any local directory. The flash drive is a form factor, not a requirement.

### Is my data really private?

Yes. FlashAI:
- Processes everything locally
- Encrypts user profiles with device-specific keys
- Never sends data to external servers
- Includes no telemetry or analytics

### Can I train it on my own data?

Absolutely. FlashAI supports learning from:
- JSON, CSV, YAML, and other data formats
- Documents (PDF, DOCX, TXT)
- Code files (Python, JavaScript, etc.)
- Markdown and HTML
- Images (with OCR)
- Interactive feedback

### What's the difference between FlashAI and fine-tuning an LLM?

Traditional fine-tuning:
- Requires significant compute resources
- Modifies model weights directly
- Expensive and time-consuming

FlashAI's approach:
- Lightweight adaptation through energy landscapes
- User preference learning without full retraining
- Runs on consumer hardware
- Incremental learning from each interaction

---

*FlashAI: Your AI. Your Data. Your Control.*

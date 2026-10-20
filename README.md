# 🤖 AI Agent - Autonomous Coding Assistant

An advanced AI-powered coding assistant with autonomous capabilities, multi-model support, and a modern web interface. Built for developers who want production-quality code generation, debugging, and system design.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Features

### 🎨 Modern Web Interface
- ChatGPT-style UI with real-time streaming
- Multi-conversation management
- File upload support (images, documents)
- Dark/light theme
- Mobile responsive

### 🤖 Autonomous Agent Capabilities
- **Continuous Coding**: Auto-continuation for long implementations
- **Task Decomposition**: Breaks complex tasks into manageable subtasks
- **Iterative Generation**: Self-improvement loops for higher quality
- **Auto-Correction**: Automatic syntax error detection and fixing
- **Long-Context Streaming**: Real-time feedback during generation

### 🧠 Multiple Agent Modes
- **Claude Code**: RAW protocol (Read-Analyze-Write) with production standards
- **Codex**: Algorithmic thinking with exhaustive testing
- **Coder**: Fast, direct code changes
- **Coding**: Professional software engineering approach
- **Architect**: System design and architecture
- **Reviewer**: Code review and security analysis
- **Debugger**: Error diagnosis and fixing
- **Security**: Security-focused analysis
- **Visualization**: Data visualization and charts

### 🔌 Multi-Model Support
- **Ollama**: Local models (Llama, Mistral, Qwen, etc.)
- **Anthropic Claude**: Claude 3.5 Sonnet, Opus, Haiku
- **CommandCode AI**: Gateway to 20+ providers (DeepSeek, GPT, Gemini, etc.)
- **Custom APIs**: Any OpenAI-compatible endpoint

### 🛠️ Developer Tools
- Code analysis and refactoring
- Git integration (status, diff, commit)
- Syntax validation
- File operations with undo support
- Grep search across codebase
- AST-based code outline extraction

### 🎯 Multimodal Capabilities
- Image analysis and description
- Video frame extraction and analysis
- Audio transcription
- Document analysis (PDF, DOCX)
- Data visualization (CSV, JSON)

### 🌐 Internet Access
- Web search (DuckDuckGo)
- URL fetching
- REST API calls
- Real-time information retrieval

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/my-ai-agent.git
   cd my-ai-agent
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys:**
   
   Create `.env.local` file:
   ```bash
   ANTHROPIC_API_KEY=your-anthropic-key
   COMMANDCODE_API_KEY=your-commandcode-key
   ```

### Usage

#### Web Interface (Recommended)

Start the web panel:
```bash
# Windows
start_web_panel.bat

# Linux/Mac
./start_web_panel.sh
```

Open browser: http://localhost:5000

#### Command Line Interface

Interactive mode:
```bash
python cli.py
```

Single command:
```bash
python cli.py "Write a Python function to sort a list"
```

With specific mode:
```bash
python cli.py "Fix this bug" --mode debugger
```

#### Python API

```python
from agent import Agent
from autonomous import AutonomousAgent, AutonomousConfig

# Create agent
agent = Agent(project_path=".", mode="claude-code")

# Simple usage
response = agent.ask("Create a REST API with Flask")

# With autonomous features
config = AutonomousConfig(
    enable_auto_continuation=True,
    enable_iteration=True,
    enable_task_decomposition=True
)
auto_agent = AutonomousAgent(agent, config)
result = auto_agent.continuous_code("Build a full authentication system")
```

## 📖 Documentation

- [Web Panel Quick Start](WEB_PANEL_QUICKSTART.md) - Get started in 3 minutes
- [Web Panel Guide](WEB_PANEL_GUIDE.md) - Complete web interface documentation
- [Autonomous Features](AUTONOMOUS_FEATURES.md) - Deep dive into autonomous capabilities
- [Anthropic Setup](ANTHROPIC_SETUP.md) - Claude API configuration
- [Configuration Guide](CONFIG_GUIDE.md) - All configuration options

## 🎯 Use Cases

### Code Generation
```python
agent = Agent(mode="claude-code")
agent.ask("Create a Python web scraper with error handling")
```

### Bug Fixing
```python
agent = Agent(mode="debugger")
agent.ask("Fix this error: TypeError: 'NoneType' object is not subscriptable")
```

### Code Review
```python
agent = Agent(mode="reviewer")
agent.ask("Review this authentication code for security issues")
```

### System Design
```python
agent = Agent(mode="architect")
agent.ask("Design a scalable microservices architecture for e-commerce")
```

### Data Visualization
```python
agent = Agent(mode="visualization")
agent.ask("Create an interactive dashboard for sales data")
```

## ⚙️ Configuration

### config.yaml

```yaml
# Model Configuration
model: claude-3-5-sonnet-20241022
timeout: 300

# API Keys (use environment variables instead)
# anthropic_api_key: ""
# commandcode_api_key: ""

# Agent Modes
default_mode: claude-code

# Autonomous Features
autonomous:
  enable_auto_continuation: true
  enable_iteration: true
  enable_task_decomposition: true
  enable_self_correction: true
  enable_streaming: true
  max_continuations: 10
  max_iterations: 20

# Model Routing (optional)
routing_mode: single  # or 'hybrid'
fast_model: llama3:latest
reasoning_model: deepseek-r1:latest
```

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
COMMANDCODE_API_KEY=cmd-...

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_NUM_PREDICT=4096

# Custom Backend
CUSTOM_MODEL_URL=https://your-api.com/v1
CUSTOM_MODEL_KEY=your-key

# Agent Behavior
AI_AGENT_REASONING=true
```

## 🏗️ Architecture

```
my-ai-agent/
├── agent.py              # Main agent logic
├── autonomous.py         # Autonomous features
├── models.py            # Multi-model backend
├── cli.py               # Command-line interface
├── webui_server.py      # Web panel backend
├── tools.py             # Developer tools
├── media.py             # Multimodal capabilities
├── memory.py            # Conversation memory
├── hooks.py             # Event hooks
├── config.py            # Configuration management
├── sessions.py          # Session management
├── panel/               # Web UI frontend
│   ├── index.html
│   ├── index.js
│   └── styles.css
├── config.yaml          # Configuration file
└── requirements.txt     # Python dependencies
```

## 🔧 Advanced Features

### Custom Agent Modes

Add your own modes in `agent.py`:

```python
ROLE_PROMPTS = {
    "my-custom-mode": (
        "Current mode: my-custom-mode. "
        "Your custom instructions here..."
    ),
}
```

### Hook System

Create automated workflows:

```python
from hooks import HookManager

hooks = HookManager()

# Run tests on file save
@hooks.on_file_save("*.py")
def run_tests(filepath):
    return {"command": "pytest"}
```

### MCP Integration

Connect to Model Context Protocol servers:

```bash
# Install MCP server
uvx install your-mcp-server

# Use in agent
agent.use_mcp_tool("server_name", "tool_name", args)
```

### Memory System

Persistent conversation memory:

```python
from memory import AgentMemory

memory = AgentMemory()
memory.remember("user-preference", "User prefers TypeScript")
preference = memory.recall("user-preference")
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=. tests/
```

Specific test:
```bash
pytest tests/test_agent.py::test_code_generation
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linters
ruff check .
black .

# Run tests
pytest
```

## 📋 Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection (for non-local models)
- Modern web browser (for web interface)

### Optional:
- Ollama (for local models)
- ffmpeg (for video analysis)
- Git (for version control features)

## 🐛 Troubleshooting

### "Response ended prematurely"
- Check internet connection
- Verify API keys
- Increase timeout in config
- Try a different model

### "Module not found"
- Reinstall dependencies: `pip install -r requirements.txt`
- Activate virtual environment
- Check Python version

### Web panel won't start
- Check port 5000 is available
- Install Flask: `pip install flask flask-cors`
- Check firewall settings

See [Troubleshooting Guide](TROUBLESHOOTING.md) for more solutions.

## 📊 Performance

- **Response time**: 1-30 seconds (depends on model and task complexity)
- **Throughput**: 100+ tokens/second (local models)
- **Memory usage**: 2-8GB (depends on model size)
- **Concurrent sessions**: 10+ (depends on hardware)

## 🔐 Security

- API keys stored in environment variables (never in git)
- Input validation and sanitization
- CORS protection
- Rate limiting support
- No code execution without user confirmation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude API
- [Ollama](https://ollama.ai) for local model support
- [CommandCode AI](https://commandcode.ai) for multi-provider gateway
- The open-source community

## 📞 Support

- **Documentation**: See `docs/` folder
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: support@example.com

## 🗺️ Roadmap

- [ ] Voice input/output
- [ ] Plugin system
- [ ] Multi-user support
- [ ] Cloud deployment guides
- [ ] Docker containerization
- [ ] VSCode extension
- [ ] Jupyter notebook integration
- [ ] Team collaboration features
- [ ] Advanced memory system
- [ ] Custom model fine-tuning

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

**Built with ❤️ for developers by developers**

[Website](https://example.com) • [Documentation](https://docs.example.com) • [Blog](https://blog.example.com) • [Twitter](https://twitter.com/example)
"# my-agent-pri" 

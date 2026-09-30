# Local Python AI Code Helper

This project provides a **Python command-line AI agent** that runs against a
**local LLM** (via [Ollama](https://ollama.com)) and helps you understand,
write, and modify code in a target project directory.

It now mixes three ideas:

- A local Ollama model as the brain.
- A Python tool-using agent loop for files, patches, and shell commands.
- Claude-style `SKILL.md` folders as reusable task expertise.

## Features

- Interactive terminal REPL with a coding assistant.
- Uses a **local model** via Ollama (for tool calling, prefer models such as
  `llama3.1` or a recent coding model that supports tools).
- Automatically loads relevant skills from `skills-main/skills`.
- Supports sticky skills with `/use <skill>`.
- Supports multiple agent modes:
  - `coder` – Quick, focused code changes
  - `coding` – Professional developer mode with tests and best practices
  - `visualization` – Creates charts, diagrams, and visual representations
  - `reviewer` – Code review and quality analysis
  - `debugger` – Error diagnosis and fixing
  - `architect` – System design and refactoring
  - `doc-writer` – Documentation and explanations
  - `security` – Security vulnerability analysis
- Simple project-scoped tools:
  - Read files inside a project.
  - List project files.
  - Overwrite files with new content (with backups).
  - Apply simple text patches.
  - Run shell commands.
  - Fetch HTTP/HTTPS URLs from the internet.
- Skill-scoped tools:
  - List files bundled with a skill.
  - Read extra skill reference files safely.

## Requirements

- Python 3.10+.
- [Ollama](https://ollama.com) installed and running on your machine.
- A pulled model. For the full agent experience, use one that supports tool
  calling, for example:

```bash
ollama pull llama3.1
```

## Backend Support

The agent supports multiple model backends:

### Ollama (Local, Recommended)
Fast local inference without API costs. Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.1
ollama pull mistral
ollama pull neural-chat
```

### OpenAI-Compatible APIs
Use OpenAI, Azure, Together.ai, or other compatible endpoints:

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Update config.yaml
model_name: gpt-4
```

### Custom Endpoints
Point to any OpenAI-compatible API:

```bash
export CUSTOM_MODEL_URL=http://localhost:8000/v1/chat/completions
export CUSTOM_MODEL_KEY=your-api-key  # optional

# In config.yaml
model_name: "custom:my-model"
```

## Installation

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

## Configuration

You can configure the agent with a `config.yaml` file in the project root:

```yaml
model_name: llama3:latest
max_context_turns: 50
project_root: C:/path/to/your/project
skills_root: C:/path/my-ai-agent/skills-main/skills
max_iterations: 60
max_auto_skills: 5
skill_prompt_chars: 20000
command_timeout: 300
list_output_limit: 0
include_ignored_files: false
```

Environment variables (if set) override values from `config.yaml`:

- `AI_AGENT_MODEL`
- `AI_AGENT_MAX_TURNS`
- `AI_AGENT_PROJECT_ROOT`
- `AI_AGENT_SKILLS_ROOT`
- `AI_AGENT_MAX_ITERATIONS`
- `AI_AGENT_MAX_AUTO_SKILLS`
- `AI_AGENT_SKILL_PROMPT_CHARS`
- `AI_AGENT_COMMAND_TIMEOUT`
- `AI_AGENT_LIST_OUTPUT_LIMIT`
- `AI_AGENT_INCLUDE_IGNORED_FILES`

## CLI Options

```bash
python cli.py --help
```

Common options:

- `--project-path` / `-p` – Path to the project directory
- `--model` / `-m` – Model name (e.g., `llama3.1`, `gpt-4`, `custom:my-model`)
- `--max-turns` – Maximum conversation history to keep (default: 50)
- `--max-iterations` – Maximum agent loop iterations (default: 60)
- `--max-auto-skills` – Auto-matched skills per request (default: 5)
- `--skill-prompt-chars` – Max characters from each skill (default: 20000)
- `--command-timeout` – Shell command timeout in seconds (default: 300)
- `--list-output-limit` – Max files shown by `/list` (default: unlimited)
- `--include-ignored-files` – Include `.git`, `.venv`, etc. in listings

## Usage

Activate your virtual environment and run:

```bash
python cli.py --project-path C:/path/to/your/project --model llama3
```

In the REPL you can use:

- `/exit` – exit the agent.
- `/help` – show available commands.
- `/skills` – list available skills.
- `/skills <query>` – search skills by task.
- `/use <skill>` – keep a skill loaded for future messages.
- `/clear-skills` – clear manually loaded skills.
- `/auto-skills on|off` – enable or disable automatic skill matching.
- `/modes` – list available agent modes.
- `/mode <name>` – switch between modes (coder, coding, visualization, reviewer, debugger, architect, doc-writer, security).
- `/limits` – show current capacity settings.
- `/read <path>` – read a file relative to the project root.
- `/list [subdir]` – list files in the project (or in a subdirectory).
- `/write <path>` – overwrite a file; you will be prompted to enter content and finish with a line containing `EOF`.
- `/patch <path>` – apply a simple text patch by providing an old and new snippet.
- `/http <url>` – manually fetch a URL over HTTP/HTTPS.

Any other input is sent to the AI assistant as a normal conversational message.
The assistant can also fetch URLs automatically with its `fetch_url` tool when
you ask about a webpage, online documentation, APIs, or current internet
content.

## Skill workflow

The agent scans `skills_root` for folders containing `SKILL.md`. For each user
message, it scores the request against skill names and descriptions, then injects
the best matches into the model prompt.

Examples:

- "Extract tables from this PDF" automatically loads the `pdf` skill.
- "Make this dashboard look better" automatically loads `frontend-design`.
- "Create a PowerPoint from these notes" automatically loads `pptx`.

You can also force a skill to stay active:

```text
/use pdf
summarize this report.pdf
```

Modes change the agent's working style:

```text
/mode coder
build this feature quickly

/mode coding
write production-quality code with tests

/mode visualization
create a dashboard showing this data

/mode reviewer
check this project for risky bugs

/mode debugger
run the tests and fix the failure

/mode security
analyze this code for vulnerabilities
```

## Power mode

The default `config.yaml` is set for a high-capacity local agent:

- 60 agent loop iterations.
- 50 recent conversation turns.
- Up to 5 auto-matched skills per request.
- 20,000 characters from each loaded skill.
- 300 second shell command timeout.
- Unlimited `/list` output for non-ignored project files.

Set `max_context_turns: 0` if you want the agent to stop trimming chat history.
The model still has its own context window, so very long histories may make
local models slower or less reliable.

## Example workflows

### Code Generation
```text
/mode coding
build a Python function that validates email addresses with regex
```

### Visualization
```text
/mode visualization
create an interactive chart showing monthly sales trends
```

### Security Review
```text
/mode security
analyze this API endpoint for common vulnerabilities
/read src/api.py
```

### Debugging
```text
/mode debugger
run the tests and fix any failures
/run-command pytest
```

### Documentation
```text
/mode doc-writer
improve the README documentation
```



## Project Structure

```
my-ai-agent/
├── agent.py          # Core agent loop with tool execution
├── cli.py            # Interactive REPL interface
├── models.py         # Multi-backend model support (Ollama, OpenAI, Custom)
├── config.py         # Configuration loading (YAML + env vars)
├── tools.py          # File and shell command operations
├── skills.py         # Skill registry and matching
├── config.yaml       # Configuration file
├── requirements.txt  # Python dependencies
├── panel/            # Web UI components (experimental)
├── skills-main/      # Claude-style skills library
└── README.md         # This file
```

## Architecture

The agent operates in a loop:

1. **User Input** → Added to conversation history
2. **Skill Matching** → Relevant skills are automatically selected
3. **Model Chat** → LLM generates response with tool calls
4. **Tool Execution** → File reads, writes, shell commands executed
5. **Result Feedback** → Tool outputs sent back to model
6. **Repeat** → Until model produces final text response

This approach allows the model to:
- Think through multi-step problems
- Verify changes by running tests
- Refine solutions based on feedback
- Use skills for domain-specific tasks

## Contributing

Contributions are welcome! Feel free to:
- Report issues and suggest improvements
- Add new skills to the `skills-main/skills` directory
- Improve documentation and examples
- Enhance the CLI or web UI

## License

This project is provided as-is for local development and learning purposes.

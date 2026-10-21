# Autononymous

A local, unrestricted AI coding agent with two interfaces — a terminal CLI and a
browser-based web panel — backed by your choice of model (local Ollama,
Anthropic Claude, DeepSeek, OpenRouter, or the multi-provider CommandCode AI
gateway). Beyond plain chat it does compiler-checked writes, draft/critique/
revise reflection, best-of-N sampling, and whole-repo hierarchical retrieval —
see [Code-quality & reasoning features](#code-quality--reasoning-features).

## Quick Start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run it
start_cli.bat          # terminal CLI
start_web_panel.bat    # browser web panel (http://localhost:5000)
```

On macOS/Linux, use `start_cli.sh` / `start_web_panel.sh` (`chmod +x` them first).

**Don't double-click `cli.py` or `web\server.py` directly** — without a venv
active they'll run under whatever Python is system-associated with `.py`
files, which usually lacks the dependencies and vanishes the console window
the instant it crashes. The `start_*` scripts activate the venv, install
missing dependencies, and keep the window open.

### Use `python -m pip`, not bare `pip`

If you ever see `Fatal error in launcher: Unable to create process...`, your
venv's `pip.exe` has a broken embedded path (common after renaming or moving
the project folder). Reinstall with:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`-m pip` goes through `python.exe` directly and ignores `pip.exe`'s broken
launcher stub entirely.

## Two interfaces

### CLI (`cli.py`)

An interactive terminal REPL. Run `start_cli.bat`, then type `/help` for the
full command list. Highlights:

| Category | Commands |
|---|---|
| Files & code | `/read /write /patch /delete /grep /outline /lines /syntax /diff /git /undo` |
| Reasoning | `/refine [rounds=N] <prompt>` (answer, then critique and revise), `/best [n=N] <prompt>` (sample N, keep the strongest) — see below |
| Memory (self-improvement) | `/memory /remember /reflect /forget /compact` — the agent keeps lessons across sessions in `.agent_memory.json` |
| Planning | `/plan /approve /todo /task` — plan mode, edit approval, a task list |
| Sessions | `/save /resume /go /sessions /new /rename /delete-session`, `/export-sft [path]` (export sessions as SFT training data) — conversations persist to `.agent_sessions/` |
| Model & routing | `/model /models /routing /reasoning` |
| Autonomy | `/loop <goal>` (runs until done or max cycles), `/agent <task>` (spawns a subagent), `/continuous /iterate /autonomous /stream /autocorrect` |
| Media | `/image /audio /video /media /data` |
| Misc | `/upload /paste @file !command #note /hooks /mcp /status /review /commit` |

Tab-completes slash commands and keeps prompt history across sessions
(`pyreadline3` on Windows, stdlib `readline` elsewhere). **Ctrl-C during a
reply stops just that generation** — whatever text had already streamed is
kept, not lost — instead of killing the whole REPL.

### Web Panel (`web/server.py`)

A Flask server with a browser chat UI (`web/index.html`, `index.js`,
`styles.css`). Run `start_web_panel.bat`, open `http://localhost:5000`.
Supports chat, streaming, file upload, mode/model switching, and the
autonomous task runner, over a JSON/SSE API (`/api/chat`, `/api/chat/stream`,
`/api/upload`, `/api/autonomous/<command>`, `/api/routing`, `/api/clear`,
`/api/modes`, `/api/models`, ...).

- **Stop button** cancels an in-flight reply (streaming keeps whatever text
  already arrived; autonomous mode has no partial output to keep, so it just
  stops).
- **Settings → Routing Mode**: switch between a single model and hybrid
  (separate fast/reasoning model pickers) live, without restarting the
  server — writes through `/api/routing`.
- **Theme toggle** (sidebar moon/sun icon) flips light/dark, saved in
  `localStorage`. **Sidebar toggle** collapses the sidebar away on desktop;
  on narrow/mobile widths it's the usual off-canvas slide-in instead.
- When the agent needs you to choose between options, it can end a reply
  with a fenced ` ```ask ` (single choice, click to send) or ` ```ask-multi `
  (checkboxes + Submit, sent as the chosen options joined with `, `) block —
  see `SYSTEM_PROMPT` in `agent.py` for the exact format.
- "New chat" / switching conversations clears the server's conversation
  state (`/api/clear`) so old context, including anything the model said
  about itself earlier, doesn't bleed into the next conversation. Resuming
  an older conversation does **not** yet replay its history back into the
  live agent — only the newest conversation's turns are ever live on the
  server at once.

> There is a second, separate web backend in this repo, `webui.py`
> (Starlette-based). It is currently **not usable** — its frontend files
> (`web/app.js`, `web/style.css`) were overwritten when the Flask panel's
> files were added to the same `web/` directory under different names. Use
> `web/server.py` via `start_web_panel.bat`, not `webui.py`, until that's
> sorted out.

## Model backends

Set `model_name` in `config.yaml`:

| Backend | `model_name` example | Key |
|---|---|---|
| Ollama (local, default) | `llama3.1:8b` | none |
| Anthropic Claude | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| CommandCode AI (gateway to DeepSeek, Kimi, GPT, GLM, Qwen, Gemini, Grok, Mistral, ...) | `commandcode:deepseek/deepseek-v4-pro` | `COMMANDCODE_API_KEY` |
| DeepSeek (direct) | `deepseek:deepseek-chat` or `deepseek:deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| OpenRouter (470+ models) | `openrouter:anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| Custom OpenAI-compatible endpoint | any name, with `custom_model_url` set | `CUSTOM_MODEL_KEY` |

> `claude-*` model ids routed through **CommandCode** only work via its
> `/messages` endpoint, not `/chat/completions` — they'll 400 through this
> gateway. Use `claude-*` directly (Anthropic Claude row above) instead.

**Hybrid routing**: set `routing_mode: hybrid` plus `fast_model` and
`reasoning_model` in `config.yaml` to automatically route short/simple
messages to a fast model and long/complex ones (keyword- and length-based
heuristic, no extra classifier call) to a stronger one. See `/routing` in
the CLI or the routing dropdowns in the web panel.

### API keys — never in `config.yaml`

`config.yaml` is git-tracked. Put keys in **`.env.local`** instead (same
directory, already in `.gitignore`, loaded automatically on startup):

```
COMMANDCODE_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
```

A value already set in your shell environment always takes precedence over
`.env.local`.

## Code-quality & reasoning features

These run inside the normal agent loop (`agent.py`) in both interfaces, no
separate setup:

- **Compiler-in-the-loop** (`verify.py`) — right after `write_file` or
  `apply_patch`, the written file is syntax-checked (Python compiled to
  bytecode in memory, JSON/YAML parsed, JavaScript via `node --check`; none
  of it executes the file) and any error is fed back to the model on the
  next step. `Agent.compiler_in_loop` (default `True`) turns it off.
- **Reflection** (`reasoning.py`, `/refine` in the CLI) — draft, critique,
  revise, repeated until the critic approves or the round budget runs out.
- **Best-of-N** (`reasoning.py`, `/best` in the CLI) — sample several
  independent answers and have the model pick the strongest; text-only, no
  tool calls during sampling.
- **Hierarchical retrieval** (`retrieval.py`, the agent's own
  `search_codebase` tool) — directories are ranked first by the vocabulary
  of the files they contain, then only files inside the best directories are
  ranked individually, each hit with a real one-line summary (docstring,
  headings, or top-level symbols with line numbers). No embeddings, rebuilds
  itself only when files actually change.
- **Context alignment** (`Agent.context_alignment`, default `True`) — before
  each model call, the files most relevant to your latest message (via the
  same index) are appended as grounding context, so long tool-calling turns
  stay anchored to what the code actually does.
- **SFT export** (`sft_export.py`, `/export-sft [path]`) — turns saved
  sessions into a standard chat-SFT JSONL file (`{"messages": [...]}` per
  line) for an actual external fine-tuning pipeline. Tool-call turns and
  placeholder/error replies are filtered out; this project doesn't train
  models itself.

## Configuration

Everything lives in `config.yaml`. Key sections:

- **Model & routing**: `model_name`, `routing_mode`, `fast_model`,
  `reasoning_model`, `reasoning_enabled`
- **Limits**: `max_context_turns`, `max_iterations`, `command_timeout`,
  `request_timeout`
- **Memory**: `memory_path`, `memory_max_lessons`, `compact_threshold`
- **Autonomous features**: `enable_auto_continuation`, `enable_iteration`,
  `enable_loops`, `enable_task_decomposition`, `enable_self_correction`,
  `enable_streaming`, plus their `max_*` / timeout counterparts
- **MCP servers**: `mcp_servers` — stdio or remote tool servers the agent
  connects to at startup (filesystem access is enabled by default)

## Project layout

```
my-ai-agent/
├── agent.py              # Core agent loop, tool dispatch, hooks/memory wiring
├── cli.py                # Terminal REPL (start_cli.bat/.sh)
├── webui.py               # Starlette web backend — currently broken, see above
├── web/
│   ├── server.py          # Flask web backend (start_web_panel.bat/.sh) — the working one
│   ├── index.html, index.js, styles.css
├── autonomous.py         # /continuous, /iterate, /autonomous, /stream, /autocorrect
├── models.py              # Ollama / Anthropic / DeepSeek / OpenRouter / CommandCode / custom backends, hybrid routing
├── verify.py               # Compiler-in-the-loop syntax checks
├── reasoning.py             # Reflection loop + best-of-N
├── retrieval.py              # Hierarchical codebase search + map-reduce summarization
├── sft_export.py              # Saved sessions -> chat-SFT JSONL
├── config.py / config.yaml / .env.local   # Configuration + secrets
├── tools.py                # File, git, shell, grep/outline tools
├── memory.py                # Persistent cross-session memory
├── sessions.py               # Save/resume conversations
├── hooks.py                   # PreToolUse/PostToolUse/UserPromptSubmit/Stop hooks
├── skills.py                   # Claude-style SKILL.md loading
├── media.py                     # Image/audio/video analysis
├── mcp_client.py                 # MCP server connections
├── diffutil.py                    # Shared diff computation
├── skills/                         # Bundled skill definitions
├── LICENSE                          # MIT
└── requirements.txt
```

## Troubleshooting

**`[WinError 5] Access is denied` on `.agent_sessions\*.json.tmp`** — a
transient Windows file lock (antivirus scan, or two agent processes saving
the same session at once). Session saves retry automatically with backoff;
if it persists, check for and close duplicate running `cli.py`/`web/server.py`
instances (`tasklist /FI "IMAGENAME eq python.exe"`).

**`Fatal error in launcher`** — see "Use `python -m pip`, not bare `pip`"
above.

**`402 Payment Required` from Ollama** — you selected a `:cloud`-suffixed
model (e.g. `deepseek-v4-pro:cloud`), which routes through Ollama's paid
hosted tier, not your local install. Switch to a model you've actually
pulled (`ollama list`) or to CommandCode AI.

**Web panel: `ModuleNotFoundError: No module named 'flask'`** — your venv's
dependencies didn't install; run
`.venv\Scripts\python.exe -m pip install -r requirements.txt` manually.

## License

[MIT](LICENSE)

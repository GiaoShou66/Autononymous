"""
Web Panel Backend Server
Connects the modern web UI to the AI agent
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import sys
import os
import re
import uuid
from pathlib import Path
import json
import time
from threading import Thread, Lock
from queue import Queue, Empty

# Add parent directory to path to import agent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent
from config import load_config
from models import ModelClient
from skills import SkillRegistry
from tools import ProjectTools
from memory import AgentMemory
from autonomous import AutonomousAgent, AutonomousConfig
from mcp_client import MCPClientManager
from hooks import HookManager
from media import analyze_media, encode_image_base64, get_image_info

# Attachments the "+" / paperclip buttons can upload inline into a message.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
MEDIA_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mkv", ".mov", ".webm"}
UPLOAD_MAX_CHARS = 15_000

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Global agent instance
agent = None
autonomous = None
cfg = None

# Pending ask_user_question() prompts: id -> Queue the frontend's answer is
# delivered through. A streaming request's worker thread blocks on its
# queue.get() until /api/chat/answer posts the user's picks.
_pending_questions: dict[str, Queue] = {}
_pending_questions_lock = Lock()

# `agent` is one global instance shared by every request — agent.ask()
# mutates agent.history (and the per-call _approval_cb/_ask_question_cb)
# with no isolation between requests. Two concurrent agent.ask() calls can
# interleave their appends (e.g. a second message arriving while the first
# is blocked inside on_ask_question waiting on the user), which corrupts
# the tool_calls/tool-result pairing in history and surfaces later as a
# confusing 400 from the backend ("tool message must follow tool_calls").
# This lock serializes agent.ask() across requests so that can't happen.
# It does NOT cover /api/chat/answer, which only delivers an answer into a
# Queue and never touches agent.history, so it can't deadlock against a
# blocked agent.ask() call holding this lock.
_agent_lock = Lock()


def _format_question_answers(answers) -> str:
    """Render the frontend's answer list as the tool-result text the model reads."""
    if not isinstance(answers, list) or not answers:
        return "No questions had options; nothing was asked."
    parts = []
    for item in answers:
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question") or "")
        picks = item.get("picks")
        if isinstance(picks, list) and picks:
            answer_text = ", ".join(str(p) for p in picks)
        else:
            answer_text = "(skipped — no preference given)"
        parts.append(f"Q: {question_text}\nA: {answer_text}")
    return "\n\n".join(parts) if parts else "No questions had options; nothing was asked."

def initialize_agent():
    """Initialize the AI agent"""
    global agent, autonomous, cfg
    
    cfg = load_config()
    
    tools = ProjectTools(
        project_root=cfg.project_root,
        command_timeout=cfg.command_timeout,
        include_ignored_files=cfg.include_ignored_files,
    )
    
    model_client = ModelClient(
        model_name=cfg.model_name,
        routing_mode=cfg.routing_mode,
        fast_model=cfg.fast_model,
        reasoning_model=cfg.reasoning_model,
    )
    skill_registry = SkillRegistry(cfg.skills_root) if cfg.skills_root else None
    
    memory_path = cfg.memory_path or (cfg.project_root / ".agent_memory.json")
    agent_memory = AgentMemory.load(memory_path, max_lessons=cfg.memory_max_lessons)
    
    settings_path = cfg.project_root / ".agent" / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
    else:
        settings = {"hooks": []}
    
    hook_manager = HookManager(settings["hooks"], cfg.project_root)
    
    agent = Agent(
        model=model_client,
        tools=tools,
        skills=skill_registry,
        mode=cfg.mode,
        max_context_turns=cfg.max_context_turns,
        max_iterations=cfg.max_iterations,
        max_auto_skills=cfg.max_auto_skills,
        skill_prompt_chars=cfg.skill_prompt_chars,
        mcp=MCPClientManager(cfg.mcp_servers, timeout=cfg.request_timeout),
        memory=agent_memory,
        compact_threshold=cfg.compact_threshold,
        hooks=hook_manager,
    )
    
    # Initialize autonomous agent
    autonomous_config = AutonomousConfig(
        enable_auto_continuation=True,
        enable_iteration=True,
        enable_loops=True,
        enable_task_decomposition=True,
        enable_self_correction=True,
        enable_streaming=True,
    )
    autonomous = AutonomousAgent(agent, autonomous_config)
    
    print(f"✓ Agent initialized with model: {cfg.model_name}")
    print(f"✓ Project root: {cfg.project_root}")
    print(f"✓ Mode: {cfg.mode}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    try:
        return send_from_directory('.', path)
    except Exception as e:
        print(f"Error serving {path}: {e}")
        return f"File not found: {path}", 404

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    global agent
    
    try:
        data = request.json
        message = data.get('message', '')
        mode = data.get('mode', 'claude-code')
        history = data.get('history', [])
        settings = data.get('settings', {})
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Switch mode if needed
        if mode != agent.mode:
            try:
                agent.set_mode(mode)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

        # Get response from agent
        try:
            with _agent_lock:
                response = agent.ask(message)
        except Exception as agent_error:
            print(f"Agent error: {agent_error}")
            import traceback
            traceback.print_exc()
            response = f"Sorry, I encountered an error: {str(agent_error)}"
        
        return jsonify({
            'response': response,
            'mode': agent.mode,
            'timestamp': time.time(),
        })
    
    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat requests"""
    global agent
    
    try:
        data = request.json
        message = data.get('message', '')
        mode = data.get('mode', 'claude-code')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Switch mode if needed
        if mode != agent.mode:
            try:
                agent.set_mode(mode)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

        def generate():
            """
            Generator for SSE streaming. agent.ask(..., on_token=...) is
            synchronous, so it runs on a worker thread; tokens (and the
            full tool-execution loop) flow back through a queue. Driving
            agent.model.chat_stream(messages) directly as if it were a
            generator was the old approach here, but chat_stream() returns
            one complete message dict, not an iterator of text chunks —
            iterating it yielded its dict *keys* ("role", "content") as
            "chunks" instead of any real model output, and skipped tool
            calls entirely.
            """
            events = Queue()

            def on_token(token):
                if token:
                    events.put({'type': 'chunk', 'content': token})

            def on_ask_question(questions):
                """
                Tool callback for ask_user_question: pushes an ask_question
                event to the browser and blocks this worker thread until
                /api/chat/answer delivers the user's picks (or a 30-minute
                cap elapses), then returns the formatted answer text as the
                tool result so the agent loop can continue.
                """
                qid = uuid.uuid4().hex
                answer_queue = Queue()
                with _pending_questions_lock:
                    _pending_questions[qid] = answer_queue
                events.put({'type': 'ask_question', 'id': qid, 'questions': questions})
                try:
                    return answer_queue.get(timeout=1800)
                except Empty:
                    return (
                        "The user did not respond to this question in time. "
                        "Make a reasonable assumption, state it, and continue."
                    )
                finally:
                    with _pending_questions_lock:
                        _pending_questions.pop(qid, None)

            def worker():
                try:
                    # Held for the whole agent.ask() call, including any
                    # time blocked inside on_ask_question waiting on the
                    # user — see _agent_lock's comment above. A second
                    # request (another message, another tab) simply waits
                    # its turn here instead of corrupting shared history.
                    with _agent_lock:
                        reply = agent.ask(message, on_token=on_token, on_ask_question=on_ask_question)
                    events.put({'type': 'done', 'reply': reply})
                except Exception as exc:
                    events.put({'type': 'error', 'message': str(exc)})
                finally:
                    events.put({'type': '__end__'})

            try:
                yield f"data: {json.dumps({'type': 'start'})}\n\n"

                Thread(target=worker, daemon=True).start()
                streamed_any = False
                final_reply = None
                while True:
                    # A short timeout lets us send SSE keepalive comments
                    # while the worker thread is blocked waiting on
                    # on_ask_question's answer_queue — otherwise an idle
                    # proxy/browser could close the connection mid-question.
                    try:
                        item = events.get(timeout=15)
                    except Empty:
                        yield ": keepalive\n\n"
                        continue
                    kind = item['type']
                    if kind == '__end__':
                        break
                    if kind == 'chunk':
                        streamed_any = True
                        yield f"data: {json.dumps(item)}\n\n"
                    elif kind == 'ask_question':
                        yield f"data: {json.dumps(item)}\n\n"
                    elif kind == 'done':
                        final_reply = item.get('reply') or ''
                    elif kind == 'error':
                        yield f"data: {json.dumps(item)}\n\n"

                # Some backends (or tool-only turns) don't stream any
                # tokens — fall back to sending the full reply as one chunk.
                if not streamed_any and final_reply:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': final_reply})}\n\n"

                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            except GeneratorExit:
                # Client disconnected
                print("Client disconnected from stream")
            except Exception as gen_error:
                print(f"Generator error: {gen_error}")
                error_data = json.dumps({
                    'type': 'error',
                    'message': str(gen_error)
                })
                yield f"data: {error_data}\n\n"
        
        response = Response(generate(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response
    
    except Exception as e:
        print(f"Error in chat_stream: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/answer', methods=['POST'])
def chat_answer():
    """
    Deliver the user's picks for a pending ask_user_question prompt (sent
    to the browser as an 'ask_question' SSE event) back to the worker
    thread blocked in on_ask_question(), unblocking it so the agent's
    tool-calling loop can continue.

    Body: {"id": "<question id from the SSE event>",
           "answers": [{"question": str, "picks": [str, ...] | null}, ...]}
    """
    try:
        data = request.json or {}
        qid = data.get('id')
        answers = data.get('answers')

        with _pending_questions_lock:
            answer_queue = _pending_questions.get(qid)

        if not qid or answer_queue is None:
            return jsonify({'error': 'Unknown or already-answered question id'}), 404

        answer_queue.put(_format_question_answers(answers))
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in chat_answer: {e}")
        return jsonify({'error': str(e)}), 500

def _ingest_upload(path: Path) -> str:
    """Turn an uploaded file into the text block injected into the chat."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        try:
            b64, mime = encode_image_base64(str(path))
            desc = agent.model.describe_image(
                image_base64=b64, prompt="Describe this image in detail.", mime_type=mime
            )
            meta = get_image_info(str(path))
            return f"[image] {meta}\n\nAI description:\n{desc}"
        except Exception as exc:
            return f"[image] {path.name} (vision analysis failed: {exc})"
    if ext in MEDIA_EXTS:
        try:
            return f"[media]\n{analyze_media(str(path))}"
        except Exception as exc:
            return f"[media] {path.name} (analysis failed: {exc})"
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[binary file] {path.name} ({path.stat().st_size} bytes) — not a text file; cannot inline."
    except OSError as exc:
        return f"[error reading {path.name}: {exc}]"
    truncated = ""
    if len(body) > UPLOAD_MAX_CHARS:
        truncated = f"\n...[truncated, {len(body) - UPLOAD_MAX_CHARS} more chars]"
        body = body[:UPLOAD_MAX_CHARS]
    return f"```\n{body}{truncated}\n```"

@app.route('/api/upload', methods=['POST'])
def upload():
    """Handle a file attachment from the "+" / paperclip buttons."""
    global agent, cfg

    uploaded_file = request.files.get('file')
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({'error': 'No file provided'}), 400

    try:
        safe_name = re.sub(r"[^\w.\-]+", "_", uploaded_file.filename) or "upload"
        uploads_dir = cfg.project_root / '.uploads'
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = uploads_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        uploaded_file.save(dest)

        block = _ingest_upload(dest)
        return jsonify({
            'filename': safe_name,
            'block': f"[Uploaded file: {safe_name}]\n{block}",
        })
    except Exception as e:
        print(f"Error in upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/autonomous/<command>', methods=['POST'])
def autonomous_command(command):
    """Handle autonomous commands"""
    global agent, autonomous

    try:
        data = request.json
        task = data.get('task', '')
        mode = data.get('mode')

        if not task:
            return jsonify({'error': 'No task provided'}), 400

        if mode and mode != agent.mode:
            try:
                agent.set_mode(mode)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

        if command == 'continuous':
            progress = []
            
            def on_progress(msg):
                progress.append(msg)
            
            result = autonomous.continuous_code(task, on_progress=on_progress)
            
            return jsonify({
                'result': result,
                'progress': progress,
            })
        
        elif command == 'iterate':
            iterations = []
            
            def on_iteration(iteration, result, score):
                iterations.append({
                    'iteration': iteration,
                    'score': score,
                    'length': len(result),
                })
            
            result = autonomous.iterative_generate(task, on_iteration=on_iteration)
            
            return jsonify({
                'result': result,
                'iterations': iterations,
            })
        
        elif command == 'decompose':
            completed_tasks = []
            
            def on_task_complete(task_obj):
                completed_tasks.append({
                    'id': task_obj.id,
                    'description': task_obj.description,
                    'status': task_obj.status.value,
                })
            
            result = autonomous.decompose_and_execute(task, on_task_complete=on_task_complete)
            
            return jsonify({
                'result': result['summary'],
                'tasks': completed_tasks,
            })
        
        else:
            return jsonify({'error': f'Unknown command: {command}'}), 400
    
    except Exception as e:
        print(f"Error in autonomous command: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/modes', methods=['GET'])
def get_modes():
    """Get available agent modes"""
    global agent
    
    try:
        return jsonify({
            'modes': agent.available_modes(),
            'current': agent.mode,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set agent mode"""
    global agent
    
    try:
        data = request.json
        mode = data.get('mode')
        
        if not mode:
            return jsonify({'error': 'No mode provided'}), 400

        try:
            agent.set_mode(mode)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        return jsonify({
            'mode': agent.mode,
            'success': True,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/skills', methods=['GET'])
def get_skills():
    """Get available skills"""
    global agent
    
    try:
        if not agent.skills:
            return jsonify({'skills': []})
        
        skills = [
            {
                'name': skill.name,
                'description': skill.description,
                'keywords': skill.keywords,
            }
            for skill in agent.skills.all()
        ]
        
        return jsonify({'skills': skills})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory', methods=['GET'])
def get_memory():
    """Get agent memory"""
    global agent
    
    try:
        return jsonify({
            'lessons': agent.memory.lessons,
            'rolling_summary': agent.memory.rolling_summary,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/changes', methods=['GET'])
def get_changes():
    """Get file changes made by agent"""
    global agent
    
    try:
        changes = [
            {
                'action': entry.action,
                'path': entry.path,
                'details': entry.details,
                'timestamp': entry.timestamp,
            }
            for entry in agent.tools.changes.entries
        ]
        
        return jsonify({'changes': changes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'agent_initialized': agent is not None,
        'model': cfg.model_name if cfg else None,
        'mode': agent.mode if agent else None,
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    global agent

    try:
        if hasattr(agent.model, 'list_available_models'):
            models = agent.model.list_available_models()
        else:
            models = [cfg.model_name]

        return jsonify({
            'models': models,
            'current': agent.model.model_name,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model', methods=['POST'])
def set_model():
    """Switch the active chat model"""
    global agent

    try:
        data = request.json
        model_name = (data.get('model') or '').strip()

        if not model_name:
            return jsonify({'error': 'No model provided'}), 400

        agent.model.model_name = model_name

        return jsonify({
            'model': agent.model.model_name,
            'success': True,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    global cfg
    
    try:
        return jsonify({
            'model': agent.model.model_name if agent else cfg.model_name,
            'mode': cfg.mode,
            'max_iterations': cfg.max_iterations,
            'max_context_turns': cfg.max_context_turns,
            'project_root': str(cfg.project_root),
            'routing_mode': agent.model.routing_mode if agent else cfg.routing_mode,
            'fast_model': agent.model.fast_model if agent else cfg.fast_model,
            'reasoning_model': agent.model.reasoning_model if agent else cfg.reasoning_model,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """
    Reset the shared agent's conversation history. There is only one
    global `agent` instance for the whole server process; the sidebar's
    separate "conversations" are a frontend/localStorage concept only
    and were never reflected server-side, so switching or creating a
    conversation left old history (and anything the model had said
    about itself) bleeding into the next one. Call this whenever the
    frontend starts or switches to a conversation.
    """
    global agent

    try:
        dropped = agent.clear_history()
        return jsonify({'success': True, 'dropped': dropped})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/routing', methods=['POST'])
def set_routing():
    """Switch between single-model, hybrid (fast/reasoning), and relay (reasoning drafts, fast finalizes) routing."""
    global agent

    try:
        data = request.json or {}
        routing_mode = (data.get('routing_mode') or '').strip().lower()

        if routing_mode not in ('single', 'hybrid', 'relay'):
            return jsonify({'error': 'routing_mode must be "single", "hybrid", or "relay"'}), 400

        if routing_mode == 'single':
            model_name = (data.get('model') or '').strip()
            if not model_name:
                return jsonify({'error': 'No model provided'}), 400
            agent.model.routing_mode = 'single'
            agent.model.model_name = model_name
        else:
            fast_model = (data.get('fast_model') or '').strip()
            reasoning_model = (data.get('reasoning_model') or '').strip()
            if not fast_model or not reasoning_model:
                return jsonify({'error': f'{routing_mode} mode needs both fast_model and reasoning_model'}), 400
            agent.model.routing_mode = routing_mode
            agent.model.fast_model = fast_model
            agent.model.reasoning_model = reasoning_model

        return jsonify({
            'success': True,
            'routing_mode': agent.model.routing_mode,
            'model': agent.model.model_name,
            'fast_model': agent.model.fast_model,
            'reasoning_model': agent.model.reasoning_model,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    print(f"404 error: {request.path}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory('.', 'index.html')

def main():
    """Main entry point"""
    print("=" * 60)
    print("AI Agent Web Panel")
    print("=" * 60)
    
    # Initialize agent
    print("\nInitializing agent...")
    initialize_agent()
    
    # Start server
    print("\n" + "=" * 60)
    print("Server starting...")
    print(f"Web Panel: http://localhost:5000")
    print(f"API: http://localhost:5000/api")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()

"""
Autonomous Agent Features:
- Continuous coding with auto-continuation
- Agentic workflow with task decomposition
- Long-context streaming
- Iterative generation and self-correction
- Auto-iteration and loop execution
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from enum import Enum


class TaskStatus(Enum):
    """Status of autonomous tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Individual task in autonomous workflow"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class AutonomousConfig:
    """Configuration for autonomous agent behavior"""
    
    # Auto-continuation
    enable_auto_continuation: bool = True
    continuation_threshold: int = 3000  # Continue if response > N chars
    max_continuations: int = 10
    
    # Iterative generation
    enable_iteration: bool = True
    max_iterations: int = 20
    iteration_improvement_threshold: float = 0.1  # Min improvement to continue
    
    # Loop execution
    enable_loops: bool = True
    max_loop_cycles: int = 100
    loop_timeout_seconds: int = 3600  # 1 hour max
    
    # Task decomposition
    enable_task_decomposition: bool = True
    max_subtasks: int = 50
    
    # Self-correction
    enable_self_correction: bool = True
    verification_on_edit: bool = True
    auto_fix_errors: bool = True
    
    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 100
    
    # Quality enhancement (Claude/Codex-level)
    enable_quality_checks: bool = True
    enforce_tests: bool = True
    enforce_documentation: bool = True
    enforce_type_hints: bool = True
    enforce_error_handling: bool = True
    min_code_quality_score: float = 0.8  # 0-1 scale


class AutonomousAgent:
    """
    Autonomous agent with continuous coding, auto-continuation,
    and iterative generation capabilities.
    """
    
    def __init__(
        self,
        base_agent,
        config: AutonomousConfig = None,
    ):
        self.base_agent = base_agent
        self.config = config or AutonomousConfig()
        
        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.task_history: List[Dict] = []
        
        # Iteration tracking
        self.iteration_count = 0
        self.iteration_results: List[str] = []
        
        # Loop tracking
        self.loop_count = 0
        self.loop_start_time = 0
        
        # Streaming buffer
        self.stream_buffer = []
    
    def continuous_code(
        self,
        initial_prompt: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Continuous coding with auto-continuation.
        
        Automatically continues generating code until the task is complete,
        handling long responses and complex implementations.
        """
        if on_progress:
            on_progress(f"[Autonomous] Starting continuous coding session")
        
        full_response = ""
        continuation_count = 0
        
        # Initial generation
        response = self.base_agent.ask(initial_prompt)
        full_response += response
        
        if on_progress:
            on_progress(f"[Continuation 0] Generated {len(response)} chars")
        
        # Auto-continuation loop
        while (
            self.config.enable_auto_continuation
            and continuation_count < self.config.max_continuations
        ):
            # Check if continuation is needed
            needs_continuation = self._needs_continuation(response)
            
            if not needs_continuation:
                break
            
            continuation_count += 1
            
            # Generate continuation
            continuation_prompt = self._build_continuation_prompt(response)
            response = self.base_agent.ask(continuation_prompt)
            full_response += "\n" + response
            
            if on_progress:
                on_progress(
                    f"[Continuation {continuation_count}] "
                    f"Generated {len(response)} chars"
                )
        
        if on_progress:
            on_progress(
                f"[Autonomous] Completed with {continuation_count} continuations, "
                f"total {len(full_response)} chars"
            )
        
        return full_response
    
    def iterative_generate(
        self,
        prompt: str,
        quality_check: Optional[Callable[[str], float]] = None,
        on_iteration: Optional[Callable[[int, str, float], None]] = None,
    ) -> str:
        """
        Iterative generation with self-improvement.
        
        Generates output, evaluates quality, and iteratively improves
        until quality threshold is met or max iterations reached.
        """
        if not self.config.enable_iteration:
            return self.base_agent.ask(prompt)
        
        best_result = ""
        best_score = 0.0
        
        for iteration in range(self.config.max_iterations):
            self.iteration_count = iteration + 1
            
            # Generate
            if iteration == 0:
                result = self.base_agent.ask(prompt)
            else:
                # Iterative improvement
                improve_prompt = self._build_improvement_prompt(
                    prompt, best_result, best_score
                )
                result = self.base_agent.ask(improve_prompt)
            
            # Evaluate quality
            if quality_check:
                score = quality_check(result)
            else:
                score = self._default_quality_check(result)
            
            if on_iteration:
                on_iteration(iteration, result, score)
            
            # Track improvement
            improvement = score - best_score
            
            if score > best_score:
                best_result = result
                best_score = score
            
            self.iteration_results.append(result)
            
            # Check if we should continue
            if iteration > 0 and improvement < self.config.iteration_improvement_threshold:
                break
            
            if score >= 0.95:  # Near-perfect score
                break
        
        return best_result
    
    def execute_loop(
        self,
        goal: str,
        condition: Optional[Callable[[], bool]] = None,
        on_cycle: Optional[Callable[[int, str], None]] = None,
    ) -> List[str]:
        """
        Autonomous loop execution until goal is achieved.
        
        Repeatedly executes actions toward a goal, with automatic
        termination when goal is reached or timeout occurs.
        """
        if not self.config.enable_loops:
            return [self.base_agent.ask(goal)]
        
        results = []
        self.loop_start_time = time.time()
        
        for cycle in range(self.config.max_loop_cycles):
            self.loop_count = cycle + 1
            
            # Check timeout
            elapsed = time.time() - self.loop_start_time
            if elapsed > self.config.loop_timeout_seconds:
                break
            
            # Check completion condition
            if condition and condition():
                break
            
            # Execute cycle
            cycle_prompt = self._build_loop_prompt(goal, cycle, results)
            result = self.base_agent.ask(cycle_prompt)
            results.append(result)
            
            if on_cycle:
                on_cycle(cycle, result)
            
            # Check if goal is achieved (heuristic)
            if self._goal_achieved(goal, result):
                break
        
        return results
    
    def decompose_and_execute(
        self,
        complex_task: str,
        on_task_complete: Optional[Callable[[Task], None]] = None,
    ) -> Dict[str, Any]:
        """
        Agentic workflow with task decomposition.
        
        Breaks down complex tasks into subtasks, executes them
        in dependency order, and aggregates results.
        """
        if not self.config.enable_task_decomposition:
            return {"result": self.base_agent.ask(complex_task)}
        
        # Decompose task
        subtasks = self._decompose_task(complex_task)
        
        # Execute in dependency order
        results = {}
        
        while True:
            # Find ready tasks
            ready_tasks = [
                task for task in subtasks
                if task.status == TaskStatus.PENDING
                and all(
                    self.tasks.get(dep, Task("", "")).status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
            ]
            
            if not ready_tasks:
                # Check if all complete or blocked
                if all(
                    task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                    for task in subtasks
                ):
                    break
                
                # Mark remaining as blocked
                for task in subtasks:
                    if task.status == TaskStatus.PENDING:
                        task.status = TaskStatus.BLOCKED
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                task.status = TaskStatus.IN_PROGRESS
                
                try:
                    # Execute task
                    task_result = self.base_agent.ask(
                        self._build_task_prompt(task, results)
                    )
                    
                    task.result = task_result
                    task.status = TaskStatus.COMPLETED
                    results[task.id] = task_result
                    
                    if on_task_complete:
                        on_task_complete(task)
                
                except Exception as e:
                    task.error = str(e)
                    task.attempts += 1
                    
                    if task.attempts >= task.max_attempts:
                        task.status = TaskStatus.FAILED
                    else:
                        task.status = TaskStatus.PENDING
        
        return {
            "tasks": subtasks,
            "results": results,
            "summary": self._summarize_results(subtasks, results),
        }
    
    def stream_generate(
        self,
        prompt: str,
        on_chunk: Callable[[str], None],
    ) -> str:
        """
        Long-context streaming generation.
        
        Generates response with streaming output for immediate feedback
        during long-running generations.
        """
        if not self.config.enable_streaming:
            return self.base_agent.ask(prompt)
        
        # Use streaming if model supports it
        if hasattr(self.base_agent.model, 'chat_stream'):
            full_response = ""
            
            def handle_delta(delta: str):
                nonlocal full_response
                full_response += delta
                on_chunk(delta)
            
            messages = self.base_agent._messages_for_model([])
            messages.append({"role": "user", "content": prompt})
            
            self.base_agent.model.chat_stream(
                messages,
                on_delta=handle_delta
            )
            
            return full_response
        else:
            # Fallback: simulate streaming
            result = self.base_agent.ask(prompt)
            
            # Stream in chunks
            for i in range(0, len(result), self.config.stream_chunk_size):
                chunk = result[i:i + self.config.stream_chunk_size]
                on_chunk(chunk)
                time.sleep(0.01)  # Small delay for visual effect
            
            return result
    
    def auto_correct(
        self,
        code: str,
        language: str = "python",
        max_attempts: int = 5,
    ) -> str:
        """
        Automatic error correction and self-improvement.
        
        Validates code, identifies errors, and automatically fixes them
        through iterative refinement.
        """
        if not self.config.enable_self_correction:
            return code
        
        current_code = code
        
        for attempt in range(max_attempts):
            # Validate code
            errors = self._validate_code(current_code, language)
            
            if not errors:
                return current_code
            
            # Generate fix
            fix_prompt = self._build_fix_prompt(current_code, errors, language)
            fixed_code = self.base_agent.ask(fix_prompt)
            
            # Extract code from response
            current_code = self._extract_code(fixed_code, language)
        
        return current_code
    
    # Helper methods
    
    def _needs_continuation(self, response: str) -> bool:
        """Check if response needs continuation"""
        # Check length threshold
        if len(response) < self.config.continuation_threshold:
            return False
        
        # Check for truncation indicators
        truncation_indicators = [
            "...",
            "[truncated]",
            "[continued in next message]",
            "# ... rest of the code",
        ]
        
        response_lower = response.lower()
        if any(ind in response_lower for ind in truncation_indicators):
            return True
        
        # Check if ends mid-sentence or mid-code
        last_lines = response.strip().split('\n')[-5:]
        last_text = '\n'.join(last_lines)
        
        incomplete_patterns = [
            lambda t: t.count('{') > t.count('}'),
            lambda t: t.count('[') > t.count(']'),
            lambda t: t.count('(') > t.count(')'),
            lambda t: not t.strip().endswith(('.', ')', '}', ']', ';')),
        ]
        
        return any(pattern(last_text) for pattern in incomplete_patterns)
    
    def _build_continuation_prompt(self, previous: str) -> str:
        """Build prompt for continuation"""
        return (
            f"Continue from where you left off. Previous output ended with:\n"
            f"```\n{previous[-500:]}\n```\n"
            f"Continue naturally without repeating what was already written."
        )
    
    def _build_improvement_prompt(
        self,
        original_prompt: str,
        previous_result: str,
        previous_score: float,
    ) -> str:
        """Build prompt for iterative improvement"""
        return (
            f"Original task: {original_prompt}\n\n"
            f"Previous attempt (score: {previous_score:.2f}):\n"
            f"{previous_result}\n\n"
            f"Improve the above solution. Focus on:\n"
            f"- Better code quality and readability\n"
            f"- More efficient algorithms\n"
            f"- Better error handling\n"
            f"- More comprehensive edge cases"
        )
    
    def _build_loop_prompt(
        self,
        goal: str,
        cycle: int,
        previous_results: List[str],
    ) -> str:
        """Build prompt for loop cycle"""
        context = ""
        if previous_results:
            recent = previous_results[-3:]  # Last 3 cycles
            context = "\n".join(f"Cycle {i}: {r[:200]}..." for i, r in enumerate(recent))
        
        return (
            f"Goal: {goal}\n"
            f"Cycle: {cycle + 1}\n"
            f"Previous cycles:\n{context}\n\n"
            f"Continue working toward the goal. What is the next step?"
        )
    
    def _build_task_prompt(self, task: Task, previous_results: Dict) -> str:
        """Build prompt for subtask execution"""
        context = ""
        for dep_id in task.dependencies:
            if dep_id in previous_results:
                context += f"\nDependency {dep_id}: {previous_results[dep_id][:200]}..."
        
        return (
            f"Task: {task.description}\n"
            f"Context from dependencies:{context}\n\n"
            f"Execute this task."
        )
    
    def _build_fix_prompt(self, code: str, errors: List[str], language: str) -> str:
        """Build prompt for error correction"""
        error_list = "\n".join(f"- {e}" for e in errors)
        return (
            f"Fix the following {language} code errors:\n"
            f"{error_list}\n\n"
            f"Original code:\n```{language}\n{code}\n```\n\n"
            f"Provide the corrected code."
        )
    
    def _decompose_task(self, task: str) -> List[Task]:
        """Decompose complex task into subtasks"""
        decompose_prompt = (
            f"Break down this complex task into concrete subtasks:\n"
            f"{task}\n\n"
            f"List 3-10 specific, actionable subtasks in JSON format:\n"
            f"[{{\"id\": \"1\", \"description\": \"...\", \"dependencies\": []}}]"
        )
        
        response = self.base_agent.ask(decompose_prompt)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tasks_data = json.loads(json_match.group())
                tasks = [
                    Task(
                        id=t["id"],
                        description=t["description"],
                        dependencies=t.get("dependencies", [])
                    )
                    for t in tasks_data[:self.config.max_subtasks]
                ]
                return tasks
        except:
            pass
        
        # Fallback: single task
        return [Task(id="1", description=task)]
    
    def _default_quality_check(self, result: str) -> float:
        """Default heuristic quality check"""
        score = 0.0
        
        # Length check (reasonable response)
        if 100 < len(result) < 10000:
            score += 0.2
        
        # Code presence
        if "```" in result or "def " in result or "class " in result:
            score += 0.3
        
        # Structure
        if result.count('\n') > 5:
            score += 0.2
        
        # Completeness
        if not result.strip().endswith(("...", "[incomplete]")):
            score += 0.3
        
        return min(score, 1.0)
    
    def _goal_achieved(self, goal: str, result: str) -> bool:
        """Check if goal is achieved (heuristic)"""
        success_indicators = [
            "done",
            "complete",
            "finished",
            "success",
            "achieved",
            "✓",
            "✅",
        ]
        
        result_lower = result.lower()
        return any(ind in result_lower for ind in success_indicators)
    
    def _validate_code(self, code: str, language: str) -> List[str]:
        """Validate code and return errors"""
        errors = []
        
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        
        return errors
    
    def _extract_code(self, response: str, language: str) -> str:
        """Extract code from markdown response"""
        import re
        
        # Try to find code block
        pattern = rf"```{language}\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1)
        
        # Fallback: return as-is
        return response
    
    def _summarize_results(self, tasks: List[Task], results: Dict) -> str:
        """Summarize task execution results"""
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        
        return (
            f"Executed {len(tasks)} tasks: "
            f"{completed} completed, {failed} failed"
        )

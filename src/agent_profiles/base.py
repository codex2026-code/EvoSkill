import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .openai_local_tools import OpenAILocalToolRuntime
from .sdk_config import is_claude_sdk, is_openai_sdk

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Import ClaudeAgentOptions at module level for type hints only
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions as ClaudeAgentOptionsType
else:
    ClaudeAgentOptionsType = Any

# Type alias for options that can be static or dynamically generated
# Supports both ClaudeAgentOptions and dict (for opencode)
OptionsProvider = Union[
    ClaudeAgentOptionsType,
    dict[str, Any],
    Callable[[], Union[ClaudeAgentOptionsType, dict[str, Any]]],
]


class AgentTrace(BaseModel, Generic[T]):
    """Metadata and output from an agent run."""

    # From first message (SystemMessage)
    uuid: str
    session_id: str
    model: str
    tools: list[str]

    # From last message (ResultMessage)
    duration_ms: int
    total_cost_usd: float
    num_turns: int
    usage: dict[str, Any]
    result: str
    is_error: bool

    # The validated structured output (None if parsing failed)
    output: Optional[T] = None

    # Error info when output parsing fails
    parse_error: Optional[str] = None
    raw_structured_output: Optional[Any] = None

    # Full response list for debugging
    messages: list[Any]

    class Config:
        arbitrary_types_allowed = True

    def summarize(
        self,
        head_chars: int = 60_000,
        tail_chars: int = 60_000,
    ) -> str:
        """
        Create a summary of the trace for passing to downstream agents.

        - On success: returns full trace
        - On failure (parse_error): truncates to head + tail to avoid context exhaustion

        Args:
            head_chars: Characters to keep from start (only used on failure)
            tail_chars: Characters to keep from end (only used on failure)
        """
        # Build the core info
        lines = [
            f"Model: {self.model}",
            f"Turns: {self.num_turns}",
            f"Duration: {self.duration_ms}ms",
            f"Is Error: {self.is_error}",
        ]

        if self.parse_error:
            lines.append(f"Parse Error: {self.parse_error}")

        if self.output:
            lines.append(f"Output: {self.output}")

        # Convert result to string
        result_str = str(self.result) if self.result else ""

        # Only truncate on failure
        if self.parse_error and len(result_str) > (head_chars + tail_chars):
            truncated_middle = len(result_str) - head_chars - tail_chars
            lines.append(f"\n## Result (truncated, {truncated_middle:,} chars omitted)")
            lines.append(f"### Start:\n{result_str[:head_chars]}")
            lines.append(f"\n[... {truncated_middle:,} characters truncated ...]\n")
            lines.append(f"### End:\n{result_str[-tail_chars:]}")
        else:
            lines.append(f"\n## Full Result\n{result_str}")

        return "\n".join(lines)


class Agent(Generic[T]):
    """Simple wrapper for running Claude agents.

    Args:
        options: Either a ClaudeAgentOptions instance (static) or a callable
                 that returns ClaudeAgentOptions (dynamic, called on each run).
        response_model: Pydantic model for structured output validation.
    """

    TIMEOUT_SECONDS = 1200  # 20 minutes
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 30  # seconds
    _CLAUDE_CODE_PRESET_OPENAI_FALLBACK = (
        "You are an autonomous coding agent operating in a local repository. "
        "Use tools when needed for accurate results, prefer deterministic changes, "
        "and keep outputs concise and directly actionable."
    )
    OPENAI_MAX_MESSAGE_CHARS = 450_000
    OPENAI_TOOL_OUTPUT_CHAR_CAP = 30_000

    @staticmethod
    def _int_env(name: str, default: int, minimum: int = 1) -> int:
        """Read a positive integer from env with safe fallback."""
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
            return default
        return max(minimum, value)

    @staticmethod
    def _string_or_unknown(value: Any) -> str:
        """Return a non-empty string value, falling back to 'unknown'."""
        if isinstance(value, str) and value:
            return value
        return "unknown"

    def __init__(self, options: OptionsProvider, response_model: Type[T]):
        self._options = options
        self.response_model = response_model
        self.timeout_seconds = self._int_env(
            "EVOSKILL_AGENT_TIMEOUT_SEC", self.TIMEOUT_SECONDS, minimum=30
        )
        self.max_retries = self._int_env(
            "EVOSKILL_AGENT_MAX_RETRIES", self.MAX_RETRIES, minimum=1
        )
        self.initial_backoff = self._int_env(
            "EVOSKILL_AGENT_INITIAL_BACKOFF_SEC", self.INITIAL_BACKOFF, minimum=1
        )

    def _get_options(self) -> Union[ClaudeAgentOptionsType, dict[str, Any]]:
        """Get options, calling the provider if it's a callable."""
        if callable(self._options):
            return self._options()
        return self._options

    @staticmethod
    def _safe_json_loads(raw_text: str) -> Any:
        """Parse JSON from text; tolerate fenced blocks and leading/trailing prose."""
        text = raw_text.strip()
        if not text:
            raise json.JSONDecodeError("Empty response", raw_text, 0)

        if "```" in text:
            parts = text.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate:
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _normalize_options_for_openai(self, options: Any) -> dict[str, Any]:
        """Convert Claude/OpenCode options to a generic OpenAI-compatible dict."""
        if isinstance(options, dict):
            normalized = dict(options)
            # Also support Claude-style keys in dict-based options.
            system_prompt = normalized.get("system_prompt")
            if isinstance(system_prompt, dict):
                if system_prompt.get("append"):
                    normalized["system"] = system_prompt["append"]
                if system_prompt.get("preset"):
                    normalized["system_preset"] = system_prompt["preset"]
            elif isinstance(system_prompt, str):
                normalized["system"] = system_prompt
            return normalized

        normalized: dict[str, Any] = {}

        system_prompt = getattr(options, "system_prompt", None)
        if isinstance(system_prompt, dict):
            if system_prompt.get("append"):
                normalized["system"] = system_prompt["append"]
            if system_prompt.get("preset"):
                normalized["system_preset"] = system_prompt["preset"]
        elif isinstance(system_prompt, str):
            normalized["system"] = system_prompt

        output_format = getattr(options, "output_format", None)
        if output_format is not None:
            normalized["format"] = output_format

        model = getattr(options, "model", None)
        if model:
            normalized["model_id"] = model

        allowed_tools = getattr(options, "allowed_tools", None)
        if allowed_tools is not None:
            normalized["allowed_tools"] = list(allowed_tools)

        cwd = getattr(options, "cwd", None)
        if cwd is not None:
            normalized["cwd"] = str(cwd)

        add_dirs = getattr(options, "add_dirs", None)
        if add_dirs is not None:
            normalized["add_dirs"] = [str(p) for p in add_dirs]

        return normalized

    def _build_openai_system_prompt(self, normalized_options: dict[str, Any]) -> str:
        """Build final OpenAI system prompt from normalized options."""
        explicit_system = str(normalized_options.get("system", "")).strip()
        preset = str(normalized_options.get("system_preset", "")).strip()

        if preset == "claude_code":
            if explicit_system:
                return (
                    f"{self._CLAUDE_CODE_PRESET_OPENAI_FALLBACK}\n\n"
                    f"Task-specific instructions:\n{explicit_system}"
                )
            return self._CLAUDE_CODE_PRESET_OPENAI_FALLBACK

        return explicit_system

    def _compact_openai_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reduce message payload size to avoid oversized context errors."""
        compacted: list[dict[str, Any]] = []
        for msg in messages:
            item = dict(msg)
            content = item.get("content")
            if isinstance(content, str) and item.get("role") == "tool":
                if len(content) > self.OPENAI_TOOL_OUTPUT_CHAR_CAP:
                    omitted = len(content) - self.OPENAI_TOOL_OUTPUT_CHAR_CAP
                    item["content"] = (
                        f"{content[:self.OPENAI_TOOL_OUTPUT_CHAR_CAP]}\n"
                        f"\n[... tool output truncated, omitted {omitted:,} characters ...]"
                    )
            compacted.append(item)

        def total_chars(payload: list[dict[str, Any]]) -> int:
            return sum(len(str(m.get("content", ""))) for m in payload)

        # Preserve system + newest messages, drop oldest non-system messages first.
        while len(compacted) > 2 and total_chars(compacted) > self.OPENAI_MAX_MESSAGE_CHARS:
            drop_idx = 1 if compacted and compacted[0].get("role") == "system" else 0
            compacted.pop(drop_idx)
        return compacted

    async def _execute_query(self, query: str) -> list[Any]:
        """Execute a single query attempt."""
        options = self._get_options()

        if is_claude_sdk():
            # Claude SDK path
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            # Convert dict to ClaudeAgentOptions if needed
            if isinstance(options, dict):
                claude_opts = ClaudeAgentOptions(
                    system_prompt=options.get("system"),
                    allowed_tools=list(options.get("tools", {}).keys())
                    if options.get("tools")
                    else [],
                    output_format=options.get("format"),
                    setting_sources=["user", "project"],
                    permission_mode="acceptEdits",
                )
                if "model_id" in options and "claude" in options["model_id"].lower():
                    claude_opts.model = options["model_id"]
                options = claude_opts

            async with ClaudeSDKClient(options) as client:
                await client.query(query)
                return [msg async for msg in client.receive_response()]
        elif not is_openai_sdk():
            # OpenCode SDK path
            from opencode_ai import AsyncOpencode

            if not isinstance(options, dict):
                raise TypeError(
                    f"OpenCode SDK requires dict options, got {type(options)}"
                )

            # Start opencode server if needed
            import subprocess

            try:
                # Quick check if server is running
                client = AsyncOpencode(base_url="http://127.0.0.1:4096")
                await client.session.create(extra_body={})
            except Exception:
                # Start server
                subprocess.Popen(
                    ["opencode", "serve", "--port", "4096", "--hostname", "127.0.0.1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(2)
                client = AsyncOpencode(base_url="http://127.0.0.1:4096")

            session = await client.session.create(extra_body={})

            extra_body = {}
            if "format" in options:
                extra_body["format"] = options["format"]

            message = await client.session.chat(
                id=session.id,
                model_id=options.get("model_id", "zai-org/GLM-5"),
                provider_id=options.get("provider_id", "togetherai"),
                parts=[{"type": "text", "text": query}],
                system=options.get("system"),
                mode=options.get("mode", "build"),
                tools=options.get("tools", {}),
                extra_body=extra_body if extra_body else None,
            )

            # Return as single-item list for consistency with Claude SDK
            return [message]
        else:
            # OpenAI SDK path (supports both official OpenAI and local OpenAI-compatible endpoints)
            from openai import AsyncOpenAI

            normalized_options = self._normalize_options_for_openai(options)
            model_name = normalized_options.get("model_id") or os.getenv("OPENAI_MODEL")
            if not model_name:
                raise ValueError(
                    "OpenAI SDK requires a model. Set --model or OPENAI_MODEL."
                )

            base_url_raw = os.getenv("OPENAI_BASE_URL", "")
            api_key_raw = os.getenv("OPENAI_API_KEY", "")
            base_url = base_url_raw.strip() or None
            api_key = api_key_raw.strip()

            http_timeout_sec = self._int_env("EVOSKILL_OPENAI_HTTP_TIMEOUT_SEC", 900)
            http_max_retries = self._int_env("EVOSKILL_OPENAI_HTTP_MAX_RETRIES", 1, 0)
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=http_timeout_sec,
                max_retries=http_max_retries,
            )

            system_text = self._build_openai_system_prompt(normalized_options)
            user_text = query

            format_hint = normalized_options.get("format")
            if format_hint:
                user_text = (
                    f"{query}\n\n"
                    "You must respond with strictly valid JSON that matches this schema: "
                    f"{json.dumps(format_hint, ensure_ascii=False)}"
                )

            messages = []
            if system_text:
                messages.append({"role": "system", "content": str(system_text)})
            messages.append({"role": "user", "content": user_text})

            runtime = OpenAILocalToolRuntime.from_options(normalized_options)
            tool_defs = runtime.get_openai_tools()
            debug_openai_tools = os.getenv("EVOSKILL_OPENAI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

            max_tool_rounds = 24
            completion = None
            query_started = time.monotonic()
            for round_idx in range(1, max_tool_rounds + 1):
                messages = self._compact_openai_messages(messages)
                round_started = time.monotonic()
                if debug_openai_tools:
                    logger.info(
                        "[OPENAI][ROUND_START] round=%s/%s messages=%s model=%s",
                        round_idx,
                        max_tool_rounds,
                        len(messages),
                        model_name,
                    )
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    tool_choice="auto" if tool_defs else None,
                )
                if debug_openai_tools:
                    logger.info(
                        "[OPENAI][ROUND_DONE] round=%s latency=%.2fs",
                        round_idx,
                        time.monotonic() - round_started,
                    )
                choice = completion.choices[0] if completion.choices else None
                msg = choice.message if choice else None
                tool_calls = getattr(msg, "tool_calls", None) if msg else None
                if not msg:
                    break

                if tool_calls:
                    if debug_openai_tools:
                        logger.info(
                            "[OPENAI][TOOL_CALLS] round=%s count=%s names=%s",
                            round_idx,
                            len(tool_calls),
                            [tc.function.name for tc in tool_calls],
                        )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )

                    for tc in tool_calls:
                        tool_started = time.monotonic()
                        tool_output = await runtime.execute(
                            tc.function.name, tc.function.arguments
                        )
                        if debug_openai_tools:
                            logger.info(
                                "[OPENAI][TOOL_DONE] round=%s name=%s latency=%.2fs output_chars=%s",
                                round_idx,
                                tc.function.name,
                                time.monotonic() - tool_started,
                                len(tool_output),
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.function.name,
                                "content": tool_output,
                            }
                        )
                    continue

                break

            if completion is None:
                raise RuntimeError("OpenAI completion failed to return a response")
            if debug_openai_tools:
                logger.info(
                    "[OPENAI][QUERY_DONE] total_latency=%.2fs rounds=%s",
                    time.monotonic() - query_started,
                    round_idx,
                )

            return [completion]

    async def _run_with_retry(self, query: str) -> list[Any]:
        """Execute query with timeout and exponential backoff retry."""
        last_error: Exception | None = None
        backoff = self.initial_backoff

        for attempt in range(self.max_retries):
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    return await self._execute_query(query)
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Query timed out after {self.timeout_seconds}s"
                )
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} timed out. Retrying in {backoff}s..."
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {backoff}s..."
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff

        raise last_error if last_error else RuntimeError("All retries exhausted")

    async def run(self, query: str) -> AgentTrace[T]:
        messages = await self._run_with_retry(query)

        if is_claude_sdk():
            # Claude SDK: messages list with SystemMessage, AssistantMessage, ResultMessage
            first = messages[0]
            last = messages[-1]

            # Try to parse structured output
            output = None
            parse_error = None
            raw_structured_output = last.structured_output

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            return AgentTrace(
                uuid=self._string_or_unknown(first.data.get("uuid")),
                session_id=self._string_or_unknown(last.session_id),
                model=self._string_or_unknown(first.data.get("model")),
                tools=first.data.get("tools", []),
                duration_ms=last.duration_ms,
                total_cost_usd=last.total_cost_usd,
                num_turns=last.num_turns,
                usage=last.usage,
                result=last.result,
                is_error=last.is_error or parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )
        elif not is_openai_sdk():
            # OpenCode SDK: single AssistantMessage with extra fields
            message = messages[0]

            # Extract structured output from info dict (extra field)
            output = None
            parse_error = None
            raw_structured_output = None

            if hasattr(message, "info") and message.info:
                raw_structured_output = message.info.get("structured")

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            # Extract text from parts (extra field)
            result_text = ""
            if hasattr(message, "parts"):
                for part in message.parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        result_text += part.get("text", "")

            # Get metadata from info dict
            info = message.info if hasattr(message, "info") else {}
            usage = info.get("tokens", {}) if info else {}
            cost = info.get("cost", 0.0) if info else 0.0

            options = self._get_options()
            model_name = (
                options.get("model_id", "unknown")
                if isinstance(options, dict)
                else "unknown"
            )
            tools = (
                list(options.get("tools", {}).keys())
                if isinstance(options, dict) and options.get("tools")
                else []
            )

            return AgentTrace(
                uuid=message.session_id or "unknown",
                session_id=message.session_id or "unknown",
                model=model_name,
                tools=tools,
                duration_ms=0,
                total_cost_usd=cost,
                num_turns=1,
                usage=usage,
                result=result_text,
                is_error=parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )
        else:
            # OpenAI SDK: ChatCompletion response
            completion = messages[0]
            choice = completion.choices[0] if completion.choices else None
            message = choice.message if choice else None
            content = message.content if message else ""
            result_text = content if isinstance(content, str) else str(content)

            output = None
            parse_error = None
            raw_structured_output = None

            if result_text:
                try:
                    raw_structured_output = self._safe_json_loads(result_text)
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = "No content returned from OpenAI API"

            options = self._normalize_options_for_openai(self._get_options())
            model_name = options.get("model_id") or os.getenv("OPENAI_MODEL") or "unknown"
            tools = list(options.get("allowed_tools", []))

            usage_payload: dict[str, Any] = {}
            if getattr(completion, "usage", None):
                usage_payload = completion.usage.model_dump()

            return AgentTrace(
                uuid=self._string_or_unknown(getattr(completion, "id", None)),
                session_id=self._string_or_unknown(getattr(completion, "id", None)),
                model=model_name,
                tools=tools,
                duration_ms=0,
                total_cost_usd=0.0,
                num_turns=1,
                usage=usage_payload,
                result=result_text,
                is_error=parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )

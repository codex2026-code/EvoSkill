"""Local tool runtime for OpenAI Chat Completions tool-calling.

This module bridges Claude-style local capabilities (filesystem + shell) into
the OpenAI SDK path by exposing a compatible set of function tools and
executing them locally.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.runtime_trace import append_runtime_trace


def _safe_json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class OpenAILocalToolRuntime:
    """Executes local tools for the OpenAI SDK path."""

    allowed_tools: list[str]
    cwd: Path
    add_dirs: list[Path] = field(default_factory=list)
    todo_state: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cwd = self.cwd.resolve()
        self.add_dirs = [p.resolve() for p in self.add_dirs]
        # Always allow cwd as a root.
        self._roots = [self.cwd, *self.add_dirs]
        skills_override = (os.getenv("EVOSKILL_SKILLS_DIR") or "").strip()
        if skills_override:
            p = Path(skills_override)
            self._skills_root = p if p.is_absolute() else (self.cwd / p)
        else:
            self._skills_root = self.cwd / ".claude" / "skills"
        self._skills_root = self._skills_root.resolve()

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "OpenAILocalToolRuntime":
        cwd = Path(options.get("cwd") or os.getcwd())
        add_dirs_raw = options.get("add_dirs") or []
        add_dirs = [Path(p) if Path(p).is_absolute() else (cwd / p) for p in add_dirs_raw]
        return cls(
            allowed_tools=list(options.get("allowed_tools", [])),
            cwd=cwd,
            add_dirs=add_dirs,
        )

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI chat.completions tool schema for enabled tools."""
        all_defs: dict[str, dict[str, Any]] = {
            "Read": {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a UTF-8 text file from allowed directories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                    },
                },
            },
            "Write": {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write UTF-8 text content to a file in allowed directories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "append": {"type": "boolean"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            "Edit": {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "description": "Replace text in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            "Glob": {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by glob pattern from cwd.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "base_dir": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            "Grep": {
                "type": "function",
                "function": {
                    "name": "Grep",
                    "description": "Search text recursively in files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                            "ignore_case": {"type": "boolean"},
                            "max_matches": {"type": "integer", "minimum": 1},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            "Bash": {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command locally under cwd.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_sec": {"type": "integer", "minimum": 1},
                        },
                        "required": ["command"],
                    },
                },
            },
            "BashOutput": {
                "type": "function",
                "function": {
                    "name": "BashOutput",
                    "description": "Alias of Bash for compatibility.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_sec": {"type": "integer", "minimum": 1},
                        },
                        "required": ["command"],
                    },
                },
            },
            "TodoWrite": {
                "type": "function",
                "function": {
                    "name": "TodoWrite",
                    "description": "Store a lightweight todo list state for this run.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {"type": "object"},
                            }
                        },
                        "required": ["todos"],
                    },
                },
            },
            "Skill": {
                "type": "function",
                "function": {
                    "name": "Skill",
                    "description": "Load a local skill from .claude/skills/<name>/SKILL.md.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "input": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
            },
            "WebFetch": {
                "type": "function",
                "function": {
                    "name": "WebFetch",
                    "description": "Fetch URL content as UTF-8 text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "max_chars": {"type": "integer", "minimum": 1},
                        },
                        "required": ["url"],
                    },
                },
            },
            "WebSearch": {
                "type": "function",
                "function": {
                    "name": "WebSearch",
                    "description": "Run a web search and return top result snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1},
                        },
                        "required": ["query"],
                    },
                },
            },
        }
        return [all_defs[t] for t in self.allowed_tools if t in all_defs]

    def _resolve_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        resolved = candidate.resolve()
        for root in self._roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise PermissionError(f"Path is outside allowed roots: {raw}")

    async def execute(self, tool_name: str, raw_args: str) -> str:
        append_runtime_trace(
            "tool_call.start",
            {
                "tool_name": tool_name,
                "args_raw_preview": (raw_args or "")[:800],
            },
        )
        if tool_name not in self.allowed_tools:
            response = _safe_json_dump({"ok": False, "error": f"Tool not allowed: {tool_name}"})
            append_runtime_trace(
                "tool_call.end",
                {
                    "tool_name": tool_name,
                    "ok": False,
                    "output_preview": response[:800],
                    "output_chars": len(response),
                },
            )
            return response

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            response = _safe_json_dump({"ok": False, "error": f"Invalid tool arguments: {e}"})
            append_runtime_trace(
                "tool_call.end",
                {
                    "tool_name": tool_name,
                    "ok": False,
                    "output_preview": response[:800],
                    "output_chars": len(response),
                },
            )
            return response

        try:
            if tool_name == "Read":
                response = self._tool_read(args)
            elif tool_name == "Write":
                response = self._tool_write(args)
            elif tool_name == "Edit":
                response = self._tool_edit(args)
            elif tool_name == "Glob":
                response = self._tool_glob(args)
            elif tool_name == "Grep":
                response = self._tool_grep(args)
            elif tool_name in ("Bash", "BashOutput"):
                response = await self._tool_bash(args)
            elif tool_name == "TodoWrite":
                response = self._tool_todo_write(args)
            elif tool_name == "Skill":
                response = self._tool_skill(args)
            elif tool_name == "WebFetch":
                response = await self._tool_webfetch(args)
            elif tool_name == "WebSearch":
                response = await self._tool_websearch(args)
            else:
                response = _safe_json_dump(
                    {"ok": False, "error": f"Unsupported tool: {tool_name}"}
                )
            append_runtime_trace(
                "tool_call.end",
                {
                    "tool_name": tool_name,
                    "ok": True,
                    "output_preview": response[:800],
                    "output_chars": len(response),
                },
            )
            return response
        except Exception as e:
            response = _safe_json_dump({"ok": False, "error": f"{type(e).__name__}: {e}"})
            append_runtime_trace(
                "tool_call.end",
                {
                    "tool_name": tool_name,
                    "ok": False,
                    "output_preview": response[:800],
                    "output_chars": len(response),
                    "exception": f"{type(e).__name__}: {e}",
                },
            )
            return response

    def _tool_read(self, args: dict[str, Any]) -> str:
        path = self._resolve_path(args["path"])
        text = path.read_text(encoding="utf-8")
        start_line = max(int(args.get("start_line", 1)), 1)
        end_line = args.get("end_line")
        lines = text.splitlines()
        if end_line is None:
            selected = lines[start_line - 1 :]
        else:
            selected = lines[start_line - 1 : int(end_line)]
        return _safe_json_dump(
            {
                "ok": True,
                "path": str(path),
                "content": "\n".join(selected),
                "line_count": len(selected),
            }
        )

    def _tool_write(self, args: dict[str, Any]) -> str:
        path = self._resolve_path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if bool(args.get("append")) else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(args["content"])
        return _safe_json_dump({"ok": True, "path": str(path), "mode": mode})

    def _tool_edit(self, args: dict[str, Any]) -> str:
        path = self._resolve_path(args["path"])
        old = args["old_text"]
        new = args["new_text"]
        replace_all = bool(args.get("replace_all", False))
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return _safe_json_dump({"ok": False, "error": "old_text not found", "path": str(path)})
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return _safe_json_dump({"ok": True, "path": str(path), "replace_all": replace_all})

    def _tool_glob(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        base_dir = args.get("base_dir", ".")
        root = self._resolve_path(base_dir)
        matches = sorted([str(p) for p in root.glob(pattern)])
        return _safe_json_dump({"ok": True, "matches": matches})

    def _tool_grep(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        root_arg = args.get("path", ".")
        ignore_case = bool(args.get("ignore_case", False))
        max_matches = int(args.get("max_matches", 200))
        root = self._resolve_path(root_arg)
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern, flags=flags)
        out: list[dict[str, Any]] = []

        files = [root] if root.is_file() else list(root.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, start=1):
                if rx.search(line):
                    out.append({"path": str(f), "line": i, "text": line})
                    if len(out) >= max_matches:
                        return _safe_json_dump({"ok": True, "matches": out, "truncated": True})
        return _safe_json_dump({"ok": True, "matches": out, "truncated": False})

    async def _tool_bash(self, args: dict[str, Any]) -> str:
        command = args["command"]
        timeout_sec = int(args.get("timeout_sec", 60))

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return _safe_json_dump({"ok": False, "error": f"Command timed out after {timeout_sec}s"})
        except asyncio.CancelledError:
            # Critical cleanup path: outer caller cancellation (e.g., task timeout) can
            # otherwise leave subprocess transports pending until interpreter shutdown.
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        return _safe_json_dump(
            {
                "ok": True,
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
        )

    def _tool_todo_write(self, args: dict[str, Any]) -> str:
        todos = args.get("todos", [])
        if not isinstance(todos, list):
            return _safe_json_dump({"ok": False, "error": "todos must be a list"})
        self.todo_state = todos
        return _safe_json_dump({"ok": True, "todos_count": len(self.todo_state)})

    def _tool_skill(self, args: dict[str, Any]) -> str:
        name = (args.get("name") or "").strip()
        if not name:
            return _safe_json_dump({"ok": False, "error": "name is required"})

        # Claude skill names are directory names. Keep lookup strict and local.
        skill_dir = (self._skills_root / name).resolve()
        try:
            skill_dir.relative_to(self._skills_root.resolve())
        except ValueError:
            return _safe_json_dump({"ok": False, "error": "invalid skill name/path"})

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return _safe_json_dump(
                {"ok": False, "error": f"Skill not found: {name}", "path": str(skill_file)}
            )

        content = skill_file.read_text(encoding="utf-8")
        return _safe_json_dump(
            {
                "ok": True,
                "name": name,
                "path": str(skill_file),
                "input": args.get("input", ""),
                "content": content,
            }
        )

    async def _tool_webfetch(self, args: dict[str, Any]) -> str:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return _safe_json_dump({"ok": False, "error": "url is required"})
        max_chars = int(args.get("max_chars", 12000))

        def _fetch() -> tuple[str, str]:
            req = urllib.request.Request(
                url.strip(),
                headers={"User-Agent": "EvoSkill/1.0 (OpenAILocalToolRuntime)"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read()
                content_type = resp.headers.get("Content-Type", "")
                text = body.decode("utf-8", errors="replace")
                return text, content_type

        text, content_type = await asyncio.to_thread(_fetch)
        trimmed = text[:max_chars]
        return _safe_json_dump(
            {
                "ok": True,
                "url": url,
                "content_type": content_type,
                "content": trimmed,
                "truncated": len(text) > len(trimmed),
            }
        )

    async def _tool_websearch(self, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return _safe_json_dump({"ok": False, "error": "query is required"})
        max_results = int(args.get("max_results", 5))
        max_results = max(1, min(max_results, 10))
        q = urllib.parse.quote_plus(query.strip())
        url = f"https://duckduckgo.com/html/?q={q}"

        # Reuse WebFetch behavior for networking and decoding.
        fetch_result = json.loads(await self._tool_webfetch({"url": url, "max_chars": 250_000}))
        if not fetch_result.get("ok"):
            return _safe_json_dump(fetch_result)

        html = fetch_result.get("content", "")
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        def _strip_tags(raw: str) -> str:
            no_tags = re.sub(r"<[^>]+>", "", raw)
            return re.sub(r"\s+", " ", no_tags).strip()

        titles = list(pattern.finditer(html))
        snippets = list(snippet_pattern.finditer(html))
        results: list[dict[str, str]] = []
        for idx, match in enumerate(titles[:max_results]):
            snippet = _strip_tags(snippets[idx].group("snippet")) if idx < len(snippets) else ""
            results.append(
                {
                    "title": _strip_tags(match.group("title")),
                    "url": urllib.parse.unquote(match.group("href")),
                    "snippet": snippet,
                }
            )

        if not results:
            return _safe_json_dump(
                {
                    "ok": False,
                    "error": "No results parsed from search provider response",
                    "provider_url": url,
                }
            )

        return _safe_json_dump(
            {
                "ok": True,
                "query": query,
                "provider_url": url,
                "results": results,
            }
        )

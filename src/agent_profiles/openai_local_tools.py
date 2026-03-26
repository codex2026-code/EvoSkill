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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
                    "description": "Compatibility stub for Skill tool.",
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
                    "description": "Compatibility stub (not implemented in local runtime).",
                    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                },
            },
            "WebSearch": {
                "type": "function",
                "function": {
                    "name": "WebSearch",
                    "description": "Compatibility stub (not implemented in local runtime).",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
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
        if tool_name not in self.allowed_tools:
            return _safe_json_dump({"ok": False, "error": f"Tool not allowed: {tool_name}"})

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return _safe_json_dump({"ok": False, "error": f"Invalid tool arguments: {e}"})

        try:
            if tool_name == "Read":
                return self._tool_read(args)
            if tool_name == "Write":
                return self._tool_write(args)
            if tool_name == "Edit":
                return self._tool_edit(args)
            if tool_name == "Glob":
                return self._tool_glob(args)
            if tool_name == "Grep":
                return self._tool_grep(args)
            if tool_name in ("Bash", "BashOutput"):
                return await self._tool_bash(args)
            if tool_name == "TodoWrite":
                return self._tool_todo_write(args)
            if tool_name in ("Skill", "WebFetch", "WebSearch"):
                return _safe_json_dump(
                    {
                        "ok": False,
                        "error": f"{tool_name} is not yet implemented in OpenAI local runtime.",
                    }
                )
            return _safe_json_dump({"ok": False, "error": f"Unsupported tool: {tool_name}"})
        except Exception as e:
            return _safe_json_dump({"ok": False, "error": f"{type(e).__name__}: {e}"})

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

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _safe_json_dump({"ok": False, "error": f"Command timed out after {timeout_sec}s"})
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
